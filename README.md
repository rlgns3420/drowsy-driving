# 졸음운전 감지 시스템

웹캠 영상에서 눈 감김, 하품, 고개 움직임을 분석해 졸음 상태를 감지하는 Python 프로젝트입니다.
Python 3.11, OpenCV, MediaPipe Face Mesh를 사용합니다.

- 약 30초간 정상 자세를 수집해 개인별 기준값 설정
- 정상·경고·위험 3단계 상태 표시 및 위험 지속 시 경보음 재생
- PC 실행 지원 및 Raspberry Pi용 설정 제공

## 실행

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

시작 후 웹캠 앞에서 정면을 보고 눈을 뜬 채 입을 닫고 기준값 수집을 기다립니다.
`C`로 기준값을 다시 수집하고, `Q` 또는 `Esc`로 종료합니다.
카메라와 감지 기준은 `config.json`에서 변경할 수 있습니다.
