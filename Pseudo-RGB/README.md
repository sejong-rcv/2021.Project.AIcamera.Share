# Pseudo-RGB
- 주야간 관계없이 사용 가능한 Pseudo-RGB 기술 개발을 위해 시작된 연구로, Colorization 모델을 이용해 열화상 영상을 컬러 영상으로 변환하고자 한다.
- 기존 Colorization의 경우 영상 전체를 입력으로 하여 영상을 colorization하고자 하였으나, 이는 다수의 물체가 나오거나 물체와 배경이 뚜렷하지 않을 경우, 물체에 대한 색상이 선명치 못하고 배경색에 덮혀버리는 문제가 발생한다. 이를 해결하고자 영상 내 각 물체에 대해서 Colorization하는 방법론이 제안되었으며, 해당 방법론을 이번 연구에 베이스로 설정하였다. Instance aware Image Colorization은 총 3개의 네트워크로 구성되어 있으며, 각각 전체 영상, 물체 영역 영상, 전체와 물체 영역을 fusion한 영상을 입력으로 한다. 실험 결과는 아래에서 확인 가능하다.
## Dataloader

데이터는 아래 구조와 같이 구성되어야만 한다.

```
datas
├── train
│   ├─ RGB
│   │   ├── LEFT_000000000.jpg
│   │   └── ...
│   ├─ THER
│   │   ├── THER_000000000.jpg
│   │   └── ...
├── test
│   ├─ RGB
│   │   ├── LEFT_000000000.jpg
│   │   └── ...
│   ├─ THER
│   │   ├── THER_000000000.jpg
│   │   └── ...
```

이때 RGB의 영상명이 LEFT로 되어있는데, 이는 KAIST DATASET에서 제공되는 left RGB을 사용하기 때문이므로, 파일이름은 크게 신경 쓸 필요가 없다.

만약 Pseudo-Lidar와 동일한 데이터 폴더 구조를 가지고 있다면 아래와 같이 sum_image.py를 실행함으로써 위의 구조로 변경가능하다. 
```
python sum_image.py --train
```
--train argument를 추가 시 train data만을 불러오며, train argmuent를 제외하면 test data를 불러온다.
## Install other dependencies
```
sh scripts/install.sh
```

## train

1. 학습에는 train.py 파일을 사용하며 이때 사용되는 주요 argument는 다음과 같다. :
 - `stage`: 해당 모델은 총 3개의 network로 구성되어 있으며 학습시킬 네트워크 구간을 의미함.
 - `train_color_img_dir`: 학습할 RGB 영상의 폴더 경로
 - `train_thermal_img_dir`: 학습할 Thermal 영상의 폴더 경로
 - `name`: 실험결과가 저장되는 폴더 이름, 모델의 체크포인트가 해당 폴더에 저장됨.
 - `input_ther` : Thermal 영상을 학습시키는지, grey 영상을 학습시키는지 설정.

 그 외에 다른 argument들은 ./options/train_options.py에서 확인 가능하다.

2. scripts/prepare_train_box.sh'sL1과 scripts/train.sh's L1에 데이터 경로를 알맞게 맞춘 후 Instance Prediction을 먼저 실행한다.
```
sh scripts/prepare_train_box.sh
```
검출된 Instance box는 $DATASET_DIR 폴더에 저장된다. 

3. 전체 학습 과정을 간단하게 실행시키기 위한 명령어는 아래와 같다.
```
sh scripts/train.sh
```
학습 과정은 3단계로 진행된다.
a. 전체 영상에 대한 Colorization network(Full network)를 학습한다.
b. a에서 학습한 Full network의 checkpoint를 사용하여 instance Colorization network(Instance network)를 학습한다.
c. 마지막으로 Full network와 Instance network를 fusion한 fusion network를 학습한다.

## test
학습된 체크포인트들은 checkpoints/mask 폴더 내에 존재한다. 만약 학습하지 않고 미리 제공된 체크포인트로 평가하려면 [해당 드라이브](https://drive.google.com/file/d/1yl7UG8bGAj25aJwDtvkicr8vAFxDz-6a/view?usp=sharing)에서 체크포인트를 다운받아 checkpoints/mask 폴더 내에 저장한 후 평가한다.

아래 명령어를 실행하면 학습된 모델을 이용하여 colorization 된 영상이 생성되며 이는 $DATASET_DIR에 저장된다.
```
bash test.sh
```
정량적 결과를 확인하기 위해서 eval.py 내 predict 폴더 경로를 맞춘 후 아래 명령어를 실행한다.
```
python eval.py
```
##Demo
test_mask.sh 파일에 INPUT_DIR을 example로 변경 후 아래 명령어를 실행하면 example image에 대한 Colorization을 사용가능하다.
```
bash test.sh
```
# 평가
## Dataset
- KAIST Multispectral Dataset
KAIST Multispectral Dataset은 캠퍼스, 도심, 주거지 등 다양한 환경과 낮, 밤에 다양한 시간대에서 촬영된 데이터 셋으로, 쌍을 이루는 컬러 스테레오 영상과 열화상 영상을 제공한다. 모든 영상들은 정합이 이루어져있으며 영상 외에 3D points, annotation 등도 제공하기에 깊이 추정, 보행자 인식, 위치 인식, 컬러 추정 등 다양한 연구 분야에 활용되는 데이터 셋이다.

- Sejong Multispectral Dataset
Sejong Multispectral Dataset은 실내물류창고 내 무인지게차의 장애물 회피 연구를 위한 목적으로 촬영되었다. 촬영에 사용된 시스템은 2대의 컬러 카메라와, 2대의 열화상 카메라로 이루어져있어 쌍을 이루는 컬러 영상과 열화상 영상을 제공한다. 주로 물류창고 내 작업자 위치 인식 연구 분야에 활용된다.

## 평가 메트릭
- PSNR과 SSIM은 영상 복원 측면에서 흔하게 사용되는 평가 메트릭으로, 먼저 PSNR의 경우 생성 또는 복원된 영상의 화질에 대한 손실 정보를 평가한다. SSIM에 경우 PSNR에 달리 인간의 시각적 화질 차이를 고려하여 설계된 평가 방법으로, 영상의 휘도, 대비, 구조 3가지 측면에 대하여 품질을 평가한다. PSNR과 SSIM 모두 값이 높을수록 영상 복원이 잘 되었다고 판단한다.
LPIPS(Learned Perceptual Image Patch Similarity)란 노이즈, 블러, 압축 등과 같은 전통적인 왜곡 종류와 generator network architecture(layers, skip connection, upsampling method 등), Loss/Learning 등과 같이 CNN 기반의 왜곡등으로 이루어진 데이터 셋을 통하여 영상 패치 기반에 유사성을 학습한 모델로 영상의 복원 정도를 평가하는 메트릭이다. 이 LPIPS는 위에 언급한 PSNR과 SSIM과 달리 값이 낮을수록 원본 영상과 차이가 적다는 것을 의미한다.


## Colorization 품질 평가
## 정량적 평가
- 먼저 열화상 영상을 이용한 컬러 추정 방법의 베이스 라인 성능을 측정해보고자 흑백 영상을 컬러 영상으로 변환한 결과(Gray2RGB)와 열화상 영상을 컬러 영상으로 변환한 결과(Ther2RGB)를 영상품질 관점에서 비교 분석하였다.

- KAIST Multispectral Dataset

| 입력 영상 | 평가 영상 | PSNR↑| SSIM↑ | LPIPS↓ |
|:-----: | :-----:|:-----:|:-----: |:-----: |
| Grey | Y(grey)+CbCr(grey)|   35.0415    | 0.9692 | 0.0822 |
| Thermal | Y(thermal)+CbCr(thermal)|  27.9761  |  0.4052 |  0.5074 |

- Sejong Multispectral Dataset

| 입력 영상 | 평가 영상 | PSNR↑| SSIM↑ | LPIPS↓ |
|:-----: | :-----:| :-----:|:-----: |:-----: |
| Grey | Y(grey)+CbCr(grey)|   34.4943    | 0.9520 | 0.0875 |
| Thermal | Y(thermal)+CbCr(thermal)|  27.9214  |  0.4422 |  0.5276 |

- 위에 결과를 살펴보면, 흑백 영상을 통해 추정된 컬러 영상보다 열화상 영상을 이용하여 추정된 컬러 영상의 성능이 모든 평가 방식에서 부족한 것을 확인할 수 있다. 하지만 이는 컬러 정보로 복원시키는 과정에서 사용되는 흑백 영상과 열화상 영상의 밝기값 차이로 인해 발생한 것으로 판단되며, 실제로 컬러 영상 복원 시 열화상 영상으로 추정된 컬러 정보(CBCR(thermal))에 흑백 영상 밝기(Y(grey))를 사용할 경우 흑백 영상 밝기Y(grey)에 흑백 영상으로 추정된 컬러 정보CbCR(grey)를 합친 결과와 유사한 결과를 얻을 수 있었다.(아래 테이블 참고)

- KAIST Multispectral Dataset

| 입력 영상 | 평가 영상 | PSNR↑| SSIM↑ | LPIPS↓ |
|:-----: | :-----:| :-----:|:-----: |:-----: |
| Grey | Y(grey)+CbCr(grey)|   35.0415    | 0.9692 | 0.0822 |
| Thermal | Y(grey)+CbCr(thermal)|  34.1562  |  0.9658 |  0.1080 |

- Sejong Multispectral Dataset

| 입력 영상 | 평가 영상 | PSNR↑| SSIM↑ | LPIPS↓ |
|:-----: | :-----:| :-----:|:-----: |:-----: |
| Grey | Y(grey)+CbCr(grey)|   34.4943    | 0.9520 | 0.0875 |
| Thermal | Y(grey)+CbCr(thermal)|  32.7555  |  0.9281 |  0.1588 |

- 이는 열화상 영상을 통해 더 나은 컬러 영상을 만들기 위해서 열화상 영상의 밝기를 흑백 영상의 밝기 스타일로 변환해야 함을 의미한다.

### 정성적 결과
![그림1.png](image/그림1.png) 1행,2행은 KAIST Multispectral Dataset, 3행,4행은 Sejong Multispectral Dataset의 예시이며, 1열부터 차례대로 Grey, RGB, Y(grey)+CbCr(grey), Y(grey)+CbCr(Thermal), Y(thermal)+CbCr(Thermal), Thermal 영상에 해당한다.

- 위에 정성적 결과를 살펴보면, 5열의 Y(Thermal)+CbCr(Thermal) 영상들은 (Y(Thermal))과 (Y(Grey))의 차이로 인하여 열화상 밝기 특성이 강하게 나타나는 것을 볼 수 있으며 특히 3행, 4행의 5열 Sejong Multispectral Dataset과 같이 물체(사람)에 대한 색상 표현에서 열화상 밝기에 컬러 정보가 뚜렷하지 않게 나오는 경향성을 볼 수 있다.
- 반면 흑백 영상 밝기(Y(Grey))를 사용하는 (Y(Grey)+CbCr(Thermal))과 (Y(Grey)+CbCr(Grey)에 경우 2행 2열, 3열, 4열의 택시처럼 원본 컬러 영상과 추정된 컬러 영상이 완전히 동일하지는 않더라도 택시에 대한 컬러 정보가 자연스럽게 표현된 것으로 볼 때, 영상의 밝기 정보가 칼라 영상 구성에 중요한 요소인 것으로 판단된다. 
## Detection 검출 성능 평가
### 정량적 결과
|  | RGB| Thermal(grey pixel)2RGB | Thermal2RGB | Grey | Thermal |
|:-----: | :-----:|:-----: |:-----: |:-----: | :-----: |
| Color information    |   O    | O | O | X | X |
| mAP(%) | 95.46 |  92.74 | 62.01 | 62.81 | 24.61 |

- 위에 결과는 Sejong Multispectral Dataset을 이용한 보행자 인식률을 나타내며, RGB 영상으로 학습된 SSD(Single Shot Detector)의 사전학습 모델을 이용하여 평가를 진행하였다.
- 정량적 결과에 의하면 컬러 영상(RGB)의 보행자 인식 성능 대비 칼라 정보가 없는 흑백 영상(Grey)와 열화상 영상(Thermal)의 보행자 인식 성능이 크게 감소한 것을 확인할 수 있으며, 반대로 컬러 정보를 추정하여 만든 (Y(Grey)+CbCr(Thermal))과 (Y(Thermal)+CbCr(Thermal)에 경우 상대적으로 보행자 인식 성능이 크게 향상된 것을 볼 수 있다. 특히 (Y(Grey)+CbCr(Thermal)의 검출 성능이 원본 컬러 영상의 검출 성능과 2.72%밖에 차이나지 않으며, 이는 열화상 영상으로 추정된 컬러 정보가 원본 영상의 컬러 정보와 매우 유사함을 나타냄과 동시에 물체 검출에서 컬러 정보가 가지는 영향력이 크다는 것을 시사한다.
### 정성적 결과
![그림2.png](image/Detection.png) 컬러 추정 연구를 활용한 딥러닝 기반 물체인식기의 정성적 평가 결과로 1열부터 차례대로 Grey, RGB, (Y(Grey)+CbCr(Thermal)), (Y(Thermal)+CbCr(Thermal)), Thermal 영상을 입력으로 한 검출 영상이다.

- 위에 정성적 결과를 살펴보면, 컬러 정보의 유무에 따라 검출의 정확도가 차이나는 것을 확인할 수 있다. 배경과 보행자가 명확히 구분되는 환경에서는 컬러 정보가 없는 흑백 영상에서도 보행자 검출이 잘 되었으나, 1행과 같이 보행자 외에도 다양한 사물이 존재하는 복잡한 상황에서 컬러 정보가 없는 흑백 영상(1행1열)과 열화상 영상(1행5열)은 보행자를 잘 검출하지 못하는 반면, 컬러 정보가 존재하는 RGB(1행2열), (Y(Grey)+Y(Thermal))(1행3열), (Y(Thermal)+CbCr(Thermal))(1행4열)의 경우 보행자를 정확히 인식한다. 이는 컬러 정보가 다양한 사물이 나오는 복잡한 환경에서 물체의 구분력을 나타내는 중요한 정보인 것을 의미한다.

### Reference
- 해당 코드는 [Instance aware Colorization](https://github.com/ericsujw/InstColorization)을 베이스로 하여 작성되었음.
