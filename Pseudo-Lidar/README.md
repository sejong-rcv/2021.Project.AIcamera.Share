# Pseudo-Lidar

## Dataset

### KAIST TITS 2018

실험한 알고리즘은 학습 시에는 Stereo-pair images와 테스트 시에는 single image가 필요하다.
학습과 테스트는  KAIST TITS 2018 데이터 셋을 사용했다.
학습에는 3037 장,테스트에는 1784 장이 포함되어있다. 

## Dataloader
Dataloader는 폴더의 다음 구조를 가정합니다 ("data_dir" 에는 Kaist_data이 들어가야한다.)
왼쪽 영상의 경우 LEFT 폴더 속에있고 , 오른쪽 영상의 경우 RIGTH에 있으니 찾기 쉬울 것이다. <b>그리고 데이터를 불러오기 위한 txt 파일은 Kaist에서 제공되는 것이 아니니 ```cp``` 나 ```mv```를 이용해 txt를 옳겨줘야한다.</b>

예 ) 데이터 폴더 구조 (이 예에서는 "Kaist_data" 디렉토리 경로를 'data_dir' 로 전달해야 한다 ) :
```
data
├── Kaist_data
│   ├── training
│   │   ├── Campus
│   │   │   ├─ DEPTH
│   │   │   │   ├── DEPTH_000000000.mat
│   │   │   │   └── ...
│   │   │   ├─ LEFT
│   │   │   │   ├── LEFT_000000000.jpg
│   │   │   │   └── ...
│   │   │   ├─ RIGTH
│   │   │   │   ├── RIGHT_000000000.jpg
│   │   │   │   └── ...
│   │   │   ├─ THERMAL
│   │   │   │   ├── THERMAL_000000000.jpg
│   │   │   │   └── ...
│   │   ├── Urban
│   │   │   ├── DEPTH
│   │   │   │   ├── DEPTH_000000000.mat
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   ├── testing
│   │   ├── Campus
│   │   │   ├─ DEPTH
│   │   │   └── ...
│   │   └── ...
│   ├── txt
│   │   ├── train.txt
│   │   ├── test.txt
│   │   ├── test_depth.txt
│   └── ...
├── models
├── output

```

## train
- 학습하기 위한 간단한 실행 명령어는 train.sh 에 있으며, 데이터 경로를 맞춰주고 train.sh을 실행 시켜주면 학습이 될 것이다.
   - 이 경우 모델의 입력은 칼라영상이며 loss는 default L1 loss를 사용한다. 
```
bash train.sh
```
- 열화상 영상을 입력으로 사용하면 다음 명령어를 사용하면 된다.
```
python main_monodepth_pytorch.py  --model resnet18_md --model_path models/resnet18_md
```
- 열화상 영상을 입력으로 사용하고, Loss를 Smooth L1 loss로 변경해 학습하려명 다음과 같은 명령어를 사용하면 된다. 
```
python main_monodepth_pytorch.py  --model resnet18_md --model_path models/resnet18_md_sl1 --l_type sl1
```
- 학습과 테스트시 main_monodepth_pytorch.py 를 사용하고 학습시 argument는 다음과 같다. :
 - `data_dir`: 학습 혹은 테스트 데이터 경로
 - `val_data_dir`:  validation 데이터 경로
 - `model_path`: 모델이 저장될 경로와 어떤 모델인지 이름으로 구분
 - `output_directory`: 테스트시 Depth 영상이 저장될 경로
 - `input_height` : 입력 영상 높이
 - `input_width` : 입력 영상 넒이
 - `model`:  encoder 모델 (resnet18_md or resnet50_md or any torchvision version of Resnet (resnet18, resnet34 etc.)
 - `pretrained`: Pretrained 된 resnet 모델을 사용할 경우 사용
 - `mode`: train or test
 - `epochs`: number of epochs,
 - `learning_rate` 
 - `batch_size` 
 - `adjust_lr`: Learning-late schedular를 사용할 것이지
 - `tensor_type`:'torch.cuda.FloatTensor' or 'torch.FloatTensor'
 - `do_augmentation`:do data augmentation or not
 - `augment_parameters`:lowest and highest values for gamma, lightness and color respectively
 - `print_images` : 학습시 영상을 저장하면서 할 것인지
 - `print_weights` : 학습시 모델을 출력할 것인지
 - `input_channels`: Number of channels in input tensor (3 for RGB images)
 - `num_workers`: Number of workers to use in dataloader
 - `RGB` : 모델의 입력으로 RGB를 사용할 것인지 아니면 열화상 영상을 사용할 것인지 



## test
테스트 argument는 학습과 동일하며 테스트 하기 위한 실행 명령어는 test.sh 에 있으니 그것을 실행 시키면 된다.

```
bash test.sh
```

### Requirements
This code was tested with PyTorch 0.4.1, CUDA 9.1 and Ubuntu 16.04. Other required modules:

```
torchvision
numpy
matplotlib
```



## 정량적 평가

- 추정된 깊이 정보의 성능 평가를 위해 성능평가지표(evaluation metric)로 RMSE(Root Mean Squre Error)와 RMLSE((Root Mean Squre Logarithmic Error)를 사용 하였음
- 깊이 정보의 성능 평가를 위해 일반적으로 최대 깊이 정보를 50m 혹은 80m 로 제한하고 있음.
- 칼라 영상(RGB)과 열화상영상(Thermal) 각각을 깊이 추정 모델의 입력 영상으로 사용하여 추정된 깊이 정보의 정확도를 분석
  - 분석 결과 열화상 영상을 입력으로 사용한 결과(RMSE)는 칼라 영상을 입력으로 사용한 결과와 소폭 차이는 있으나 유사한 것으로 판단됨
  - 또한 RMSLE 에서는 오히려 열화상 영상을 입력으로한 결과가 좋은 성능을 보임. 이를 통해 전체적인 Depth 의 정확성은 칼라를 입력으로 했을 때 더욱 좋은 성능을 보장하지만, 가까운 물체의 깊이정보가 중요할 경우 열화상 영상을 입력으로 하면 좋다는 결론이 도출 된다.   

| model |  입력영상| RMSE <50| RMLSE<50m | RMSE <80m| RMSLE<80 m|
|:-----: | :-----:|:-----: |:-----: |:-----: |:-----: |
| Monodepth |   칼라  |  4.2886 |  0.2038  | 4.2886 | 0.2038 |
| Monodepth |   열화상 |  4.7079 |  0.1988 | 4.7079 | 0.1988 |

- 베이스 라인 방법론에 멀티스펙트럴 데이터셋을 적용했을 경우 문제되는 Depth Hall을 줄이기 위해 Balanced L1 loss , Smooth L1 Loss 적용 성능
  - Smooth L1 loss를 L1 loss 대신 적용했을 때 RMSLE 평가 지표에서 가장 좋은 성능을 나타냄, 또한 열화상을 입력으로 했을 때는 50m 이하 RMSE 에서도 베이스라인 성능 보다 좋은 성능을 보인다. 

| model |  입력영상| RMSE <50m | RMLSE<50m | RMSE <80m| RMSLE<80m |
|:-----: | :-----:|:-----: |:-----: |:-----: |:-----: |
| Monodepth |   칼라  |  4.2886 |  0.2038  | 4.2886 | 0.2038 |
| Monodepth+Balanced L1 loss |   칼라 |  4.4707 |  0.1880 | 5.5746 | 0.1955 |
| Monodepth+Smooth L1 loss(제안된 방법론) |   칼라  |  4.3035 |  0.1805  | 5.1633 | 0.1895 |
| Monodepth |   열화상 |  4.7079 |  0.1988 | 4.7079 | 0.1988 |
| Monodepth+Balanced L1 loss |   열화상  |  4.7470 |  0.1941  | 5.8399 | 0.2058 |
| Monodepth+Smooth L1 loss(제안된 방법론) |   열화상 |  4.4529 |  0.1830 | 5.1450 | 0.1897 |


## 정성적 평가
- 자가 학습 기반 깊이 추정 모델(Pseudo-Lidar)의 정성적 결과
   - 4열과 5열을 비교해볼 경우, 깊이 추정을 위해 사용된 입력 영상의 종류(칼라영상 혹은 열화상 영상)는 깊이 추정 결과에 작은 차이는 유발하나 매우 큰 영향은 미치지 않는 것으로 판단됨
   - 3행 5열, 7열을 보면 Depth Hall 해결을 위해 제안된 Smooth L1 Loss가 제 역할을 하는 것을 알 수 있다. 
![eval_visualize](images/eval_visualize.png) (그림 1) 왼쪽부터 칼라영상, 열화상영상, 정답 Disparity 영상, 칼라 영상을 이용한 베이스라인 결과 , 열화상영상을 이용한  베이스라인 결과, 칼라 영상을 이용한 제안된 방법론 결과, 열화상 영상을 이용한 제안한 방법론 결과
- 칼라영상, 열화상 영상을 이용하여 추정된 깊이 정보를 3D 공간상으로 투영한 정성적 결과
  - (그림 2)와 (그림 3) 모두 3D 공간 표현에 큰 차이가 없는 것을 볼 수 있다.
  - 그림 2 와 같이 열화상 영상으로 추측한 것에 칼라 정보를 입힐 경우 사람이 입체 공간을 판단하기 매우 좋은 영상을 얻을 수 있다. 이를 통해 열화상 영상으로 칼라 영상을 잘 변환 할 경우 낮과 밤 상관 없이 운전자 혹은 군인이 주변 상황을 판단하기 좋은 입체 공간을 취득할 수 있는 것 알 수 있다. 

![visualize_Thermal](images/visualize_Thermal.jpg) (그림 2) 왼쪽부터  열화상 영상 , 추정된 깊이 영상,  추정된 Pseudo-Lidar 

![visualize_Thermal](images/visualize_RGB.jpg) (그림 3) 왼쪽부터  칼라 영상 , 추정된 깊이 영상,  추정된 Pseudo-Lidar 


