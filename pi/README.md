# 라즈베리 파이 설치 및 실행 안내

대상 환경: **라즈베리 파이 4·5, 라즈베리 파이 운영체제 북웜(Bookworm) 64비트, 시스템 파이썬 3.11**.
32비트 운영체제 및 파이썬 3.13 기반 운영체제에는 이 설치 스크립트를 사용하지 않습니다.
현재 버전을 고정한 MediaPipe 얼굴 특징점 추적 기능의 ARM64용 파이썬 3.11 사전 빌드 패키지를 사용합니다.
라즈베리 파이의 초당 처리 프레임 수(FPS)·카메라·스피커 검증은 설치 후 아래 절차로 수행하세요.
소프트웨어 검증 범위는 [검증 기록](VALIDATION.md)에 기록했습니다.

## 컴퓨터에서 파일을 분리하고 옮기기

프로젝트 전체가 있는 컴퓨터에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe tools/build_pi.py
```

생성되는 `dist/drowsiness-pi.zip` 하나만 파이로 복사합니다.
압축 파일에는 `pi/`, 공통 `drowsiness/`, 설정, 필요한 패키지 목록, 파일 검증용 해시 목록인 `MANIFEST.json`이 포함됩니다.
컴퓨터용 Tkinter 화면, 가상환경, 촬영 영상, 분석 결과는 포함하지 않습니다.
공통 코드를 수정하면 ZIP을 다시 생성해야 합니다. **소스의 pi 폴더만 복사하면 실행되지 않습니다.**

라즈베리 파이에서 압축을 해제한 `drowsiness-pi` 폴더로 이동합니다.

```bash
unzip drowsiness-pi.zip
cd drowsiness-pi
bash pi/setup.sh
```

설치 시 인터넷·sudo 권한이 필요합니다. 스크립트는 시스템 패키지를 apt로 설치하고,
프로젝트의 가상환경인 `.venv`에 파이썬 패키지를 설치합니다. 운영체제나 시스템 파이썬을 교체하지 않습니다.
CSI용 Picamera2/libcamera는 apt 패키지를 공유하는 `--system-site-packages` 환경을 사용합니다.
이미 다른 파이썬 버전으로 만든 `.venv`가 있다면 새 폴더에 압축을 다시 풀어 설치하세요.

## 카메라 및 소리 연결 확인

USB UVC 웹캠:

```bash
.venv/bin/python pi/check.py --source usb --camera 0
```

CSI 리본 카메라:

```bash
rpicam-hello --list-cameras
.venv/bin/python pi/check.py --source csi --camera 0
```

결과의 수신 프레임 수(`frames_received`)가 30이면 영상 수신과 모델 실행이 성공한 것입니다.
얼굴 추적 성공 프레임 수(`tracked_frames`)가 0이면 얼굴·조명·각도를 확인합니다. 이 검사는 정확도 평가가 아닙니다.
USB 장치 번호는 `ls /dev/video*`로 확인하고 `--camera`로 변경합니다.
CSI 카메라는 기존 `raspistill` 기반의 구형 카메라 방식 대신 Picamera2를 사용합니다.

스피커 또는 USB 오디오 장치를 연결한 후 한 번의 시험음을 재생하려면:

```bash
.venv/bin/python pi/check.py --source usb --audio
```

경보는 **오디오 출력 장치**를 사용하며 GPIO 부저 제어는 포함하지 않습니다.
소리 오류는 실행 기록과 `report.json`의 오디오 오류 항목(`audio_error`)에 표시됩니다.

## 바로 실행

```bash
# 화면 없이 USB 카메라로 실행, 스피커 경보 활성
bash pi/run.sh --source usb

# CSI 카메라, 스피커 경보 활성
bash pi/run.sh --source csi

# 연결 점검 중 무음 실행
bash pi/run.sh --source usb --mute

# 모니터가 연결된 데스크톱에서 판정 화면 표시
bash pi/run.sh --source usb --preview

# 60초 실행 후 자동 종료
bash pi/run.sh --source usb --duration 60

# 파이에 복사해 둔 촬영 영상으로 동일한 코드 점검 (자동 무음)
bash pi/run.sh --video ./driver.mp4
```

시작 후 정면·눈 뜸·입 닫음 정상 자세를 **유효 관측 30초** 동안 유지해야 개인 기준값이 완성됩니다.
얼굴이 보이지 않거나 고개가 크게 돌아간 시간은 수집에서 제외됩니다.
`Ctrl+C` 또는 `SIGTERM`으로 종료하면 결과를 저장하고 카메라·모델·오디오 자원을 정리합니다.
화면 모드는 Q/Esc 또는 창 닫기로 종료합니다. 카메라가 멈춰 영상 읽기 작업이 끝나지 않는 경우에는
장치 드라이버가 응답할 때까지 종료가 지연될 수 있습니다.

상태와 초당 처리 프레임 수는 기본 5초 간격으로 실행 기록에 출력됩니다. 기본 결과 경로는
`results/pi-날짜-시간/`이며 `--output`으로 **새 폴더**를 지정할 수 있습니다.

- `frames.csv`: 경과 시간, 상태, 얼굴 추적 여부, 눈 열림 비율(EAR)·입 벌림 비율(MAR), 각도, 동작 지속 시간, 경보 요청
- `events.csv`: 기준값 수집·정상·주의·위험·추적 실패 및 경보 요청 변경 시점
- `report.json`: 처리량, 추적 비율, 개인 기준값, 적용 설정, 종료 상태

기본 실행은 촬영 영상을 디스크에 저장하지 않습니다. 실행 기록과 CSV 파일의 크기는 실행 시간에 따라 증가합니다.
`alarm_requested`는 감지 로직의 경보 요청이며 실제 스피커 출력 성공을 뜻하지 않습니다.
종료 코드: 0=실행 완료, 1=설정/장치/실행 오류, 2=개인 기준값 미완료.
파이의 영상 입력·추론·기록을 포함한 처리 속도가 너무 낮아 프레임 간격이 최대 관측 간격인 `max_observation_gap`(기본 0.5초)을
넘으면 지속 시간 근거가 초기화됩니다. 안정적으로 초당 2프레임을 넘는지 확인하고 조명·해상도·냉각을 점검하세요.

`config.pi.json`에서 추론용 영상 너비·카메라의 초당 프레임 수·경보를 조정합니다. 임계값을 생략하면 공통 기본값이 적용됩니다.
파이는 추론용 영상 너비가 320화소이고 컴퓨터 기본값은 480화소이므로 같은 영상이라도 지표가 달라질 수 있습니다.
컴퓨터에서 점검할 때 `test/video.py --config config.pi.json`으로 파이 설정을 먼저 비교할 수 있습니다.

## 설치 근거

- [MediaPipe 0.10.18 배포 파일](https://pypi.org/project/mediapipe/0.10.18/): ARM64용 파이썬 3.11 사전 빌드 패키지
- [공식 Picamera2 설명서](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf): apt 설치 및 시스템 패키지를 공유하는 가상환경 구성
- [Picamera2 공식 소스의 픽셀 포맷](https://github.com/raspberrypi/picamera2/blob/main/picamera2/request.py): RGB888의 BGR 배열

실제 사람의 졸음 검출률은 별도의 실영상 라벨과 비교해야 합니다. 촬영한 눈 감김 동작과 실제 피로의 정답은 구분하세요.
