# 졸음운전 감지 시스템

일반 USB 웹캠과 노트북 내장 카메라로 운전자의 눈 감김, 입 벌림, 고개 움직임을 분석하는 졸음운전 감지 프로그램입니다.
파이썬 3.11, OpenCV, MediaPipe의 얼굴 특징점 추적 기능을 사용합니다.

- 한국어 화면에 정상·주의·졸음 위험 판정과 감지 수치 표시
- 개인 기준값 수집, 카메라 선택, 영상 재생, 경보음 켜기·끄기
- 카메라 없이 화면과 판정 흐름을 확인하는 모의 실행 및 이식 전 검증 도구 제공

## 컴퓨터에서 실행하기

프로젝트 폴더에서 다음 명령을 실행해 가상환경과 필요한 패키지를 설치합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe test/local.py
```

`test/run-local.cmd`를 더블클릭해도 실행됩니다. 웹캠 시작 후 정면을 보고 약 30초간 기준값을 수집합니다.
컴퓨터용 점검 화면에서는 경보음이 기본으로 꺼져 있으며, 졸음 판정은 화면에 표시됩니다.

카메라 없이 실행: `.\.venv\Scripts\python.exe test/local.py --demo`

자세한 실행·검증 방법은 [컴퓨터용 점검 안내](test/README.md)를 참고하세요.
실제 졸음 감지 정확도와 사고 예방 효과는 별도 실영상 검증이 필요합니다.

## 실제 촬영 영상 점검

```powershell
.\.venv\Scripts\python.exe test/video.py --video "D:\videos\driver.mp4" --preview --realtime --save-video
```

또는 영상 파일을 `test/run-video.cmd` 위로 끌어놓으세요.
시작에 정상 자세 35초 이상이 포함된 영상을 사용합니다.
프레임별 수치, 상태 변경 시각, 요약 보고서와 판정 표시 영상이 `results/`에 저장됩니다.
정답 구간을 작성한 CSV 파일을 넣으면 오탐률·검출률도 계산합니다. [실영상 점검 안내](test/VIDEO_GUIDE.md)를 참고하세요.

## 라즈베리 파이로 옮겨 실행하기

컴퓨터에서 다음 명령을 실행하면 라즈베리 파이에 필요한 파일만 압축합니다.

```powershell
.\.venv\Scripts\python.exe tools/build_pi.py
```

`dist/drowsiness-pi.zip`을 파이에 복사·압축 해제한 뒤 실행합니다.

```bash
bash pi/setup.sh
bash pi/run.sh --source usb
# CSI 카메라: bash pi/run.sh --source csi
```

대상은 라즈베리 파이 4·5, 라즈베리 파이 운영체제 북웜(Bookworm) 64비트, 파이썬 3.11입니다.
화면 없이 실행하며 `--preview`로 화면을 열 수 있습니다.
장치 점검과 자세한 설치·실행 방법은 [라즈베리 파이 설치 안내](pi/README.md)를 참고하세요.

## 전시 모드

Windows에서 `test/run-exhibit.cmd`를 더블클릭하세요. 설치된 `.venv`를 사용하며 웹캠을 자동 시작합니다.
직접 실행: `python test/local.py --exhibit` 또는 `python main.py --ui desktop --simple`.
카메라 없이 확인: `python test/local.py --exhibit --demo` (시뮬레이션 표시, 경보음 없음).

큰 영상과 정상 / 주의 / 졸음 위험 판정을 표시합니다. C는 기준값 재수집,
F/F11은 전체화면, M은 경보음 전환, Esc는 종료입니다. 관람객 교대 시 C를 누르세요.
자동 재수집은 적용하지 않았습니다. 정상 정면 자세의 유효 관측 시간 기준 기본 10초이며,
설정 파일의 `"display": {"simple": true, "calibration_seconds": 10.0}`으로 조절합니다.
샘플 수·품질 조건도 충족해야 합니다. 기존 상세 모드는 옵션 없이 그대로 실행합니다.
카메라 오류 시 나타나는 시뮬레이션 버튼으로 시연을 계속할 수 있습니다.
전시 실행 스크립트는 경보음을 기본으로 켜며, 시뮬레이션은 소리를 재생하지 않습니다.

UI 검증: `python test/local.py --exhibit --ui-smoke`.
정상·주의·위험 판정 카드 문구를 확인하고 `.cache/exhibit-ui/normal.png`,
`danger.png`, `tracking.png`에 화면을 저장합니다 (데스크톱과 Pillow 필요).
