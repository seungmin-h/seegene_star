# seegene_challenge
## 패치 분류 모델

---

패치를 분류하기 위한 환경셋팅과 코드 사용법입니다.

### 실행 환경

현재는 아래와 같이 conda 환경 및 관련 패키지를 설치한 뒤 실행할 수 있습니다. 현재 **Ubuntu 20.04.3 LTS / python 3.8 / torch 1.8.0** 에서 테스트를 통과했습니다.
```bash
$ conda create -n myenv python=3.8 -y
$ source activate myenv
$ pip install -r requirements.txt
```
위의 명령어에서 'myenv' 부분은 환경명이므로 자유롭게 수정 가능합니다. requirements.txt 에 기재된 패키지가 모두 설치되면,
아래와 같이 torch(1.8.0 이상에서)가 GPU를 사용할 수 있는 상태가 되었는지 확인 바랍니다. 

```bash
$ source activate myenv
$ python # 파이썬 실행
$ >>> import torch
$ >>> torch.cuda.is_available() # True 가 출력되는지 확인 
$ >>> torch.backends.cudnn.enabled # True 가 출력되는지 확인 
```
True 가 출력되지 않으면, https://pytorch.org/ 로 접속하신 후, install 커맨드를 직접 찾아 재설치 해주시기 바랍니다.


### 학습 
* **데이터 절대경로 생성**

학습에 사용할 train, test 데이터셋의 절대경로를 다음 파일을 통해 생성해 주시기 바랍니다. 이때, make_abspath.py 스크립트에서
"make_abspath_txt(train 폴더의 절대경로, txt 파일이 생성될 경로)" 를 기재하신 후 아래의 커맨드를 실행 바랍니다. 
```bash
$ cd ./src/train/utils
$ python make_abspath.py
```

* **변수 설정**

위에서 생성된 txt 파일은 학습에 사용됩니다. 생성된 txt 파일의 경로를 *.yaml 에 기록합니다. 만일, txt 파일 외에 다른 파일이나 
숨김파일이 있다면 오류가 발생합니다.

* **실행**

실행 커맨드는 다음과 같습니다.
```bash
$ cd ./src/train
$ python main.py
```

### 추론

* **실행**

학습이 완료된 모델은, 학습에서 사용한 *.yaml 파일과 함께 일원관리 됩니다. 추론은 이 모델파일과 설정파일을 불러온 뒤 실행됩니다.
아래 커맨드를 실행하게 되면, 가장 최근에 학습된 모델로 추론을 바로 시작합니다. 만약, 과거에 학습된 모델로 추론을 진행하고 싶다면, weights/experiments 
이하에서 원하는 모델을 복사한 후, weights/checkpoints 에 붙여넣고 커맨드를 실행해 주시기 바랍니다.
```bash
$ cd ./src/infernce
$ python main.py
```
