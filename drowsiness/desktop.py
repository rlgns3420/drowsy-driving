"""Local Korean dashboard, using standard-library Tk and the shared pipeline."""
import base64
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import queue
import time
import tkinter as tk
from tkinter import filedialog, ttk

import cv2

from .camera import discover_cameras
from .demo import demo_sample
from .metrics import LEFT_EYE, MOUTH, RIGHT_EYE
from .pipeline import MonitoringPipeline
from .presentation import StatusView, status_view
from .worker import FramePacket, MonitorWorker


BG, CARD, BORDER = "#0b1220", "#131f30", "#26354b"
TEXT, MUTED, ACCENT = "#edf4fc", "#94a9c3", "#2dd4bf"
FONT = "Malgun Gothic"


class MonitorApp:
    def __init__(self, config, demo=False, video=None):
        self.config = config
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError("GUI를 열 수 없습니다. 데스크톱 환경에서 실행하거나 --ui opencv를 사용하세요.") from exc
        self.root.title("Drive Guard | 로컬 졸음 감지")
        self.root.geometry("1180x800+24+24")
        self.root.minsize(1040, 800)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.worker = None
        self.scanning = False
        self.closing = False
        self.last_packet_at = None
        self.photo = None
        self.scan_results = queue.Queue()
        self.video_path = Path(video) if video else None
        self.mode = tk.StringVar(value="demo" if demo else "video" if video else "camera")
        self.camera_index = tk.StringVar(value=str(config.camera.index))
        self.backend = tk.StringVar(value=config.camera.backend)
        self.audio = tk.BooleanVar(value=config.alarm.enabled)
        self._build()
        self.root.bind("<Escape>", lambda event: self.close())
        self.root.bind("<Key-c>", lambda event: self.recalibrate())
        self.root.after(30, self._poll)

    def label(self, parent, text, size=11, color=TEXT, bg=CARD, bold=False, **kwargs):
        return tk.Label(parent, text=text, bg=bg, fg=color,
                        font=(FONT, size, "bold" if bold else "normal"), **kwargs)

    def button(self, parent, text, command, primary=False):
        return tk.Button(parent, text=text, command=command, bg=ACCENT if primary else BORDER,
                         fg=BG if primary else TEXT, activebackground="#45e5cc" if primary else "#354862",
                         activeforeground=BG if primary else TEXT, relief="flat", bd=0,
                         padx=16, pady=9, cursor="hand2", font=(FONT, 10, "bold"))

    def _build(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=CARD, background=BORDER,
                        foreground=TEXT, arrowcolor=TEXT, padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", CARD)],
                  foreground=[("readonly", TEXT)])
        style.configure("Guard.Horizontal.TProgressbar", troughcolor=BORDER,
                        background=ACCENT, borderwidth=0, thickness=6)
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=24, pady=(20, 16))
        self.label(header, "DRIVE GUARD", 22, bg=BG, bold=True).pack(side="left")
        self.label(header, "LOCAL MONITOR  /  PC 검증", 10, MUTED, BG).pack(side="left", padx=18)
        self.source_label = self.label(header, "카메라 연결 대기", 10, ACCENT, BG)
        self.source_label.pack(side="right")

        toolbar = tk.Frame(self.root, bg=CARD, padx=14, pady=12)
        toolbar.pack(fill="x", padx=24)
        self.label(toolbar, "카메라 번호", 10, MUTED).pack(side="left", padx=(0, 8))
        self.index_input = tk.Spinbox(toolbar, from_=0, to=20, width=3, textvariable=self.camera_index,
                                      bg=BG, fg=TEXT, buttonbackground=BORDER, relief="flat",
                                      font=(FONT, 11), insertbackground=TEXT)
        self.index_input.pack(side="left")
        self.backend_input = ttk.Combobox(toolbar, textvariable=self.backend, width=7,
                                          values=("auto", "dshow", "msmf", "v4l2"), state="readonly")
        self.backend_input.pack(side="left", padx=10)
        self.scan_button = self.button(toolbar, "카메라 찾기", self.scan)
        self.scan_button.pack(side="left")
        self.video_button = self.button(toolbar, "영상 열기", self.select_video)
        self.video_button.pack(side="left", padx=8)
        self.demo_button = self.button(toolbar, "시뮬레이션", self.select_demo)
        self.demo_button.pack(side="left")
        self.stop_button = self.button(toolbar, "중지", self.stop)
        self.stop_button.pack(side="right", padx=(8, 0))
        self.stop_button.configure(state="disabled")
        self.start_button = self.button(toolbar, "웹캠 시작", self.start_camera, True)
        self.start_button.pack(side="right")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=16)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=316)
        body.rowconfigure(0, weight=1)
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        video_card = tk.Frame(left, bg=CARD)
        video_card.pack(fill="both", expand=True)
        video_heading = tk.Frame(video_card, bg=CARD)
        video_heading.pack(fill="x", padx=16, pady=12)
        self.label(video_heading, "실시간 영상", 11, bold=True).pack(side="left")
        self.video_badge = self.label(video_heading, "OFFLINE", 9, MUTED)
        self.video_badge.pack(side="right")
        self.video_canvas = tk.Canvas(video_card, bg="#070d17", highlightthickness=0)
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.video_canvas.bind("<Configure>", self._placeholder)
        self.perf = self.label(left, "FPS --    |    추론 -- ms    |    얼굴 추적 대기", 9, MUTED, BG, anchor="w")
        self.perf.pack(fill="x", pady=(8, 12))
        metrics_row = tk.Frame(left, bg=BG)
        metrics_row.pack(fill="x")
        self.metric_values = []
        self.metric_notes = []
        for index, name in enumerate(("눈 열림 · EAR", "입 벌림 · MAR", "고개 · P / Y / R")):
            metrics_row.columnconfigure(index, weight=1, uniform="metric")
            card = tk.Frame(metrics_row, bg=CARD, padx=12, pady=12)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 8, 0))
            self.label(card, name, 9, MUTED).pack(anchor="w")
            value = self.label(card, "--", 18 if index < 2 else 13, bold=True)
            value.pack(anchor="w", pady=(8, 4))
            note = self.label(card, "개인 기준 수집 대기", 8, MUTED)
            note.pack(anchor="w")
            self.metric_values.append(value)
            self.metric_notes.append(note)

        right = tk.Frame(body, bg=BG, width=316)
        right.grid(row=0, column=1, sticky="nsew")
        right.pack_propagate(False)
        self.status_card = tk.Frame(right, bg=CARD, padx=16, pady=8)
        self.status_card.pack(fill="x")
        self.status_eyebrow = self.label(self.status_card, "현재 판정", 10, MUTED)
        self.status_eyebrow.pack(anchor="w")
        self.status_title = self.label(self.status_card, "연결 대기", 18, bold=True, anchor="w")
        self.status_title.pack(fill="x", pady=(6, 6))
        self.status_detail = self.label(self.status_card, "카메라를 연결하고 감지 시작을 누르세요.",
                                        10, MUTED, wraplength=275, justify="left", anchor="w")
        self.status_detail.pack(fill="x")
        self.visual_alert = self.label(self.status_card, "화면 알림 항상 활성", 8, ACCENT)
        self.visual_alert.pack(anchor="w", pady=(4, 0))

        calibration = tk.Frame(right, bg=CARD, padx=16, pady=8)
        calibration.pack(fill="x", pady=10)
        self.label(calibration, "개인 기준값", 10, bold=True).pack(anchor="w")
        self.calibration_text = self.label(calibration, "정상 자세 30초 관측 후 판정 시작", 9, MUTED)
        self.calibration_text.pack(anchor="w", pady=4)
        self.progress = ttk.Progressbar(calibration, maximum=100, style="Guard.Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(0, 6))
        self.reset_button = self.button(calibration, "기준값 다시 수집  ·  C", self.recalibrate)
        self.reset_button.configure(pady=4)
        self.reset_button.pack(fill="x")
        self.reset_button.configure(state="disabled")

        evidence = tk.Frame(right, bg=CARD, padx=16, pady=10)
        evidence.pack(fill="x")
        self.label(evidence, "지속 시간", 10, bold=True).pack(anchor="w")
        self.duration_labels = []
        for title in ("눈 감김", "하품 / 입 벌림", "고개 자세 이탈"):
            row = tk.Frame(evidence, bg=CARD)
            row.pack(fill="x", pady=(4, 0))
            self.label(row, title, 10, MUTED).pack(side="left")
            value = self.label(row, "0.0 s", 10)
            value.pack(side="right")
            self.duration_labels.append(value)
        audio_check = tk.Checkbutton(evidence, text="경보음 켜기", variable=self.audio,
                                      command=self.toggle_audio, bg=CARD, fg=TEXT,
                                      activebackground=CARD, activeforeground=TEXT,
                                      selectcolor=BG, font=(FONT, 9), anchor="w")
        audio_check.pack(fill="x", pady=(6, 0))
        self.audio_text = self.label(evidence, "경보음 꺼짐 · 화면 판정은 유지", 8, MUTED, wraplength=275)
        self.audio_text.pack(anchor="w", pady=(4, 0))
        self.toggle_audio()

        history = tk.Frame(right, bg=CARD, padx=16, pady=8)
        history.pack(fill="both", expand=True, pady=(10, 0))
        self.label(history, "상태 기록", 10, bold=True).pack(anchor="w")
        self.history = tk.Text(history, height=4, width=30, wrap="word", bg=CARD, fg=MUTED,
                               relief="flat", font=(FONT, 9), state="disabled", cursor="arrow")
        self.history.pack(fill="both", expand=True, pady=(4, 0))
        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=24, pady=(0, 16))
        self.help_text = self.label(footer, "USB / 내장 웹캠 사용  ·  정면 얼굴과 충분한 조명 필요  ·  Esc 종료",
                                    9, MUTED, BG, anchor="w")
        self.help_text.pack(fill="x")

    def _placeholder(self, event=None):
        if self.photo is None:
            self.video_canvas.delete("all")
            self.video_canvas.create_text(self.video_canvas.winfo_width() // 2,
                                           self.video_canvas.winfo_height() // 2,
                                           text="카메라를 연결해주세요\n\n카메라 없이 확인하려면 시뮬레이션을 선택하세요.",
                                           fill=MUTED, font=(FONT, 13), justify="center")

    def log(self, message):
        self.history.configure(state="normal")
        self.history.insert("end", f"{datetime.now():%H:%M:%S}  {message}\n")
        if int(self.history.index("end-1c").split(".")[0]) > 100:
            self.history.delete("1.0", "2.0")
        self.history.see("end")
        self.history.configure(state="disabled")

    def _status(self, view):
        self.current_status = view.key
        self.status_card.configure(bg=view.background)
        for label in (self.status_eyebrow, self.status_title, self.status_detail, self.visual_alert):
            label.configure(bg=view.background)
        self.status_title.configure(text=view.title, fg=view.color)
        self.status_detail.configure(text=view.detail, fg=TEXT)
        self.visual_alert.configure(fg=view.color)

    def _busy(self, busy):
        for button in (self.start_button, self.scan_button, self.video_button, self.demo_button):
            button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy and self.worker is not None else "disabled")
        self.reset_button.configure(state="normal" if busy and self.worker is not None else "disabled")
        self.index_input.configure(state="disabled" if busy else "normal")
        self.backend_input.configure(state="disabled" if busy else "readonly")

    def start(self):
        if self.worker is not None or self.scanning:
            return
        config = deepcopy(self.config)
        try:
            config.camera.index = int(self.camera_index.get())
            config.camera.backend = self.backend.get()
            config.alarm.enabled = self.audio.get()
            config.validate()
        except ValueError as exc:
            self.log(f"설정 오류: {exc}")
            return
        self.worker = MonitorWorker(config, self.mode.get(), self.video_path)
        self.last_packet_at = None
        self._clear_metrics()
        self._busy(True)
        self.source_label.configure(text="시뮬레이션 · 실제 카메라 검증 아님" if self.mode.get() == "demo" else
                                    "영상 재생 준비" if self.mode.get() == "video" else "카메라 연결 중")
        self._status(StatusView("CONNECTING", "연결 중", "영상 입력과 얼굴 추적 모델을 준비하고 있습니다.",
                                "#60a5fa", "#132641"))
        self.log("감지를 시작합니다.")
        self.worker.start()

    def start_camera(self):
        self.mode.set("camera")
        self.start()

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.stop_button.configure(state="disabled")
            self.reset_button.configure(state="disabled")
            self._status(StatusView("STOPPING", "종료 중", "카메라와 모델을 정리하고 있습니다.", MUTED, CARD))

    def recalibrate(self):
        if self.worker is not None and not self.worker.stopped.is_set():
            self.worker.commands.put(("reset", None))

    def toggle_audio(self):
        enabled = self.audio.get()
        self.audio_text.configure(text="위험 지속 시 경보음 재생" if enabled else "경보음 꺼짐 · 화면 판정은 유지")
        if self.worker is not None:
            self.worker.commands.put(("audio", enabled))

    def select_demo(self):
        self.mode.set("demo")
        self.start()

    def select_video(self):
        path = filedialog.askopenfilename(title="검증할 영상 선택", filetypes=[("동영상", "*.mp4 *.avi *.mov *.mkv"),
                                                                             ("모든 파일", "*.*")])
        if path:
            self.video_path = Path(path)
            self.mode.set("video")
            self.start()

    def scan(self):
        if self.worker is not None or self.scanning:
            return
        import threading
        self.mode.set("camera")
        self.scanning = True
        self._busy(True)
        self.log("카메라 0~3번에서 실제 영상 수신을 확인합니다.")
        config = deepcopy(self.config.camera)
        config.backend = self.backend.get()

        def scan_devices():
            try:
                self.scan_results.put((discover_cameras(config), None))
            except Exception as exc:
                self.scan_results.put(([], str(exc)))
        threading.Thread(target=scan_devices, name="camera-scan", daemon=True).start()

    def _clear_metrics(self):
        self.progress["value"] = 0
        for value in self.metric_values:
            value.configure(text="--")
        for note in self.metric_notes:
            note.configure(text="개인 기준 수집 대기")
        for value in self.duration_labels:
            value.configure(text="0.0 s")
        self.calibration_text.configure(text="정상 자세 관측 후 판정 시작")
        self.perf.configure(text="FPS --    |    추론 -- ms    |    얼굴 추적 대기")

    def _show_packet(self, packet):
        observation = packet.observation
        self._status(status_view(observation))
        self.source_label.configure(text=packet.source)
        self.video_badge.configure(text="SIMULATION" if packet.source.startswith("시뮬레이션") else "LIVE / REPLAY",
                                   fg="#fbbf24" if packet.source.startswith("시뮬레이션") else ACCENT)
        self.progress["value"] = observation.progress * 100
        self.calibration_text.configure(text="기준값 설정 완료" if observation.profile else
                                         f"정상 자세 {observation.valid_seconds:.1f}초 수집 · {observation.progress * 100:.0f}%")
        metrics, profile, detection = observation.metrics, observation.profile, observation.detection
        values = (f"{metrics.ear:.3f}", f"{metrics.mar:.3f}",
                  f"{metrics.pose.pitch:+.0f} / {metrics.pose.yaw:+.0f} / {metrics.pose.roll:+.0f}°") if metrics else ("--", "--", "--")
        for label, value in zip(self.metric_values, values):
            label.configure(text=value)
        if profile:
            for label, text in zip(self.metric_notes, (f"눈 감김 기준 < {profile.ear_threshold:.3f}",
                    f"입 벌림 기준 > {profile.mar_threshold:.3f}", "개인 정면 자세 대비 각도")):
                label.configure(text=text)
        durations = (detection.eye_seconds, detection.yawn_seconds, detection.head_seconds) if detection else (0, 0, 0)
        for label, duration in zip(self.duration_labels, durations):
            label.configure(text=f"{duration:.1f} s")
        self.perf.configure(text=f"FPS {packet.fps:.1f}    |    추론 {packet.inference_ms:.1f} ms    |    " +
                           ("얼굴 추적 중" if metrics else "얼굴 추적 불가"))
        self.audio_text.configure(text=packet.audio_error or ("시뮬레이션 · 소리 재생 안 함" if packet.source.startswith("시뮬레이션") else
                                   "경보음 활성" if packet.audio_enabled else "경보음 꺼짐 · 화면 판정은 유지"))
        frame = packet.frame.copy()
        if self.config.camera.draw_landmarks and packet.landmarks is not None:
            for index in set(LEFT_EYE + RIGHT_EYE + MOUTH):
                x, y = map(round, packet.landmarks[index])
                cv2.circle(frame, (x, y), 2, (190, 215, 45), -1)
        if self.config.camera.mirror and not packet.source.startswith("시뮬레이션"):
            frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        available_w, available_h = max(1, self.video_canvas.winfo_width()), max(1, self.video_canvas.winfo_height())
        scale = min(available_w / width, available_h / height)
        frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        ok, png = cv2.imencode(".png", frame)
        if ok:
            self.photo = tk.PhotoImage(data=base64.b64encode(png))
            self.video_canvas.delete("all")
            self.video_canvas.create_image(available_w // 2, available_h // 2, image=self.photo)
            if self.current_status == "DANGER":
                self.video_canvas.create_rectangle(2, 2, available_w - 2, available_h - 2,
                                                    outline="#fb7185", width=5)
                self.video_canvas.create_text(available_w // 2, 32, text="졸음 위험 감지",
                                               fill="#fb7185", font=(FONT, 20, "bold"))

    def _poll(self):
        if self.closing:
            return
        try:
            devices, error = self.scan_results.get_nowait()
            self.scanning = False
            self._busy(False)
            if devices:
                self.camera_index.set(str(devices[0].index))
                self.log("연결 가능: " + ", ".join(f"{item.index}번 ({item.width}×{item.height})" for item in devices))
            else:
                self.log(error or "카메라 없음: USB 연결 / Windows 카메라 권한 / 다른 앱 점유를 확인하세요.")
        except queue.Empty:
            pass
        worker = self.worker
        if worker is not None:
            try:
                packet = worker.frames.get_nowait()
                if not worker.stopped.is_set():
                    self._show_packet(packet)
                    self.last_packet_at = time.monotonic()
            except queue.Empty:
                if not worker.stopped.is_set() and self.last_packet_at is not None and time.monotonic() - self.last_packet_at > 1.5:
                    self._clear_metrics()
                    self.photo = None
                    self._placeholder()
                    self.video_badge.configure(text="STALE", fg="#fbbf24")
                    self._status(StatusView("STALE", "영상 입력 지연", "최신 영상이 없습니다. 카메라 연결을 확인하세요.",
                                            "#fbbf24", "#362b17"))
            while True:
                try:
                    kind, message = worker.events.get_nowait()
                except queue.Empty:
                    break
                self.log(message)
                if kind in ("finished", "failed"):
                    self.worker = None
                    self.last_packet_at = None
                    self._busy(False)
                    self._clear_metrics()
                    self.photo = None
                    self._placeholder()
                    self.video_badge.configure(text="OFFLINE", fg=MUTED)
                    if kind == "failed":
                        self._status(StatusView("ERROR", "연결 / 처리 오류", "카메라 연결과 권한을 확인하세요.\n오류 내용은 상태 기록을 확인하세요.",
                                                "#fb7185", "#3b1b29"))
                    else:
                        self._status(StatusView("STOPPED", "감지 종료", message, MUTED, CARD))
        self.root.after(30, self._poll)

    def close(self):
        self.closing = True
        if self.worker is not None:
            self.worker.stop()
        deadline = time.monotonic() + 4

        def finish():
            if self.worker is not None and self.worker.is_alive() and time.monotonic() < deadline:
                self.root.after(50, finish)
            else:
                self.root.destroy()
        finish()

    def smoke_check(self, output):
        """Exercise actual Tk widgets with clearly labelled synthetic decisions."""
        output.mkdir(parents=True, exist_ok=True)
        self.root.update()
        config = deepcopy(self.config)
        config.calibration.seconds, config.calibration.min_samples = 1.0, 20
        pipeline = MonitoringPipeline(config)
        seen = set()
        for index in range(29 * 30):
            now = index / 30
            frame, metrics = demo_sample(now)
            observation = pipeline.update(metrics, now)
            key = status_view(observation).key
            if key in seen:
                continue
            seen.add(key)
            self.log("UI 검증 · " + status_view(observation).title)
            self._show_packet(FramePacket(frame, None, observation, 30, 0,
                              "시뮬레이션 · 실제 카메라 검증 아님", False, None))
            self.root.update()
            assert self.status_title.cget("text") == status_view(observation).title
            if key in ("NORMAL", "DANGER", "TRACKING"):
                # Pillow is optional, used only for this verification artifact.
                from PIL import ImageGrab
                self.root.update_idletasks()
                x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
                ImageGrab.grab(bbox=(x, y, x + self.root.winfo_width(),
                                    y + self.root.winfo_height())).save(output / f"{key.lower()}.png")
        assert {"CALIBRATING", "READY", "NORMAL", "WARNING", "DANGER", "TRACKING"} <= seen
        self.audio.set(False)
        self.toggle_audio()
        assert "화면 판정은 유지" in self.audio_text.cget("text")
        self._clear_metrics()
        assert self.metric_values[0].cget("text") == "--"
        self.mode.set("demo")
        self.start()

        def pump(seconds):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)

        pump(2.5)
        assert self.worker is not None and self.worker.is_alive()
        assert self.current_status in ("NORMAL", "READY")
        self.recalibrate()
        pump(0.15)
        assert self.current_status == "CALIBRATING"
        self.stop()
        pump(0.25)
        assert self.worker is None and self.current_status == "STOPPED"
        # Failure handling is deterministic; this does not probe real hardware.
        from unittest.mock import patch
        with patch("drowsiness.worker.Webcam", side_effect=RuntimeError("UI test: camera missing")):
            self.start_camera()
            pump(0.25)
        assert self.current_status == "ERROR" and self.worker is None
        assert self.start_button.cget("state") == "normal"
        self.root.geometry("1040x800+24+24")
        self.root.update()
        assert self.history.winfo_height() >= 20
        self.root.destroy()
        print("UI smoke passed: calibration, normal, warning, danger, tracking loss, audio off, "
              "worker start/stop, recalibration, camera-error recovery")
        return 0
