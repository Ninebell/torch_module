# Torch Module
torch module what I use and create.

### Layers
- Conv2D 
> Param | 기존 conv, batch, activation 분리되어있던걸 Sequential로 묶음
>- input channel: 입력 channel 크기
>- outputc channel: 출력 channel 크기
>- kernel size: 사용 커널 크기
>- stride: stride 크기
>- padding: padding 크기
>- activation: 활성화 함수
>- batch: 배치 유무

  
- AttentionBlock [Paper](https://arxiv.org/abs/1807.06521)
> Param | CBAM 구현으로 Channel, Spatial Attention이 묶여 있음
>- feature: 입력 channel 크기
>- ratio: channel attention에서 축소 비율

- BottleNeckBlock [Paper](https://arxiv.org/abs/1512.03385)
> Param | BottleNeck 구조로 Attention Block 사용 가능
>- input channel: 입력 channel 크기
>- attention: attention block 유무
>- ratio: attention block의 ratio
>- activation: 활성화 함수
- Hourglass [Paper](https://arxiv.org/abs/1603.06937)
> Param | Hourglass 구조로 각 Layer들은 BottleNeck으로 구성.
>- feature: 입력 channel 크기
>- layers: hourglass 깊이
>- attention: attention block 유무
>- ratio: attention block의 ratio
- ProjLayer(need to fix)
> Non Param | 3D Point를 직교 정사영 시키는 Layer
- DenseBlock [Paper](https://arxiv.org/abs/1608.06993)
> Param | DenseNet에서 사용되는 Block
>- input channel: 입력 channel 크기
>- growth ch: Layer 커지는 비율
>- layer: Layer 수
>- activation: 활성화 함수

- UPConv2D
> Param | Upsample을 시행 후 Conv2D 수행
>- up scale: Upsampling에 사용되는 scale 값
>- input channel: 입력 channel 크기
>- output channel: 출력 channel 크기
>- kernel size: 사용 커널 크기
>- stride: stride 크기
>- padding: padding 크기
>- activation: 활성화 함수
>- batch: 배치 유무

### Utils

- TorchBoard
> Param | tensor board
>- dir path: tensor board 저장 경로
>- comment: tensor board comment

- train model
> Param | 모델 학습 기본 틀
>- epoches: 반복 회수
>- model: 학습 시킬 모델
>- loss: 사용 할 손실 함수
>- optim: 사용 할 옵티마이저
>- train_loader: 학습 데이터 로더
>- validate_loader: 검증 데이터 로더
>- save_path: model, tensor board 저장 경로
>- tag: tensor board tag
>- checkpoint: 
>   >model, train loss, validate loss를  이용한 check point 
>- accuracy
>   >y, predict 값을 이용한 accuracy 계산 함수

- get param count
> Param | 모델 총 파라미터 수
>- net: 모델