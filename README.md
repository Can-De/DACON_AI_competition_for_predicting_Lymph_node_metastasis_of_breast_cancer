# AI competition for predicting Lymph node metastasis of breast cancer ([DACON_유방암의 임파선 전이 예측 AI 경진대회](https://dacon.io/competitions/official/236011/overview/description))

## [배경]

- 림프절(임파선)은 암의 전이, 암이 퍼지는 데 매우 치명적인 역할을 합니다.

  병원에서 암 진단을 받았을 때 가장 많이 듣는 말이자 우리가 관심을 가져야 하는 것이 림프절 전이이며, 

  이러한 림프절 전이 여부에 따라 치료와 예후가 달라집니다.

  따라서 림프절 전이 여부와 전이 단계를 파악하는 것이 암을 치료하고 진단하는 것에 있어서 매우 핵심적인 역할을 합니다.

- 이번 '유방암의 임파선 전이 예측 AI경진대회'에서 유방암 병리 슬라이드 영상과 임상 항목 데이터를 이용하여,

  유방암 치료에 핵심적인 역할을 할 수 있는 최적의 AI 모델을 만들어 유방암의 임파선 전이 여부를 예측해 보고자 합니다.

## [주최 / 주관]

- 주최 : 연세대학교 의과대학, JLK, MTS
- 후원 : 보건산업진흥원
- 주관 : 데이콘

## [모델링]

- Attention-based multiple instance-learning (AMIL) 방식 적용
  - 여러 개의 인스턴스(샘플)로 이루어진 가방(bag) 데이터 셋트를 분류하는 머신러닝 알고리즘입니다.
  - 가방 데이터 세트의 인스턴스중 일부는 양(positive) 클래스에 속하고, 일부 인스턴스는 음(negative) 클래스에 속합니다.
  - 가방의 각 인스턴스를 독립적으로 분류하는 것이 아니라, 전체 가방 데이터 세트를 분류합니다.
  - MIL은 의학 분야를 비롯한 다양한 분야에서 사용되는 유용한 머신러닝 기술 중 하나입니다.
- 적용 조건
  - 각 Patch의 크기는 128*128 픽셀로 Crop합니다.
  - GRAYSCALE 값이 240 이하인 펙셀의 비율이 70% 이상인 Patch만 사용합니다.
  - 각 가방에는 10개의 Patch를 담습니다.
  - 각 환자의 최소 가방 수는 10개로 부족한 Patch는 해당 환자의 Patch를 증식하여 사용합니다.

## [구현 세부 정보]
데이터 준비
각 가방의 patch 갯수(N)는 10으로 고정되어 있지만, 각 WSI의 가방 번호(M)는 고정되어 있지 않으며 WSI의 해상도에 따라 달라집니다. 통계 결과에 따르면 WSI의 백 번호(M)는 1에서 300까지 다양하며, 훈련 및 테스트 중에 WSI에 대해 고정되지 않습니다. 데이터 세트 준비 과정은 다음 그림과 같으며, 자세한 내용은 다음과 같습니다:
<div align="center">
    <img src="imgs/a.png" alt="c"/>
</div>

먼저 각 WSI에 대해 주석이 달린 종양 영역을 잘라내고, 한 WSI에 여러 개의 주석이 달린 종양 영역이 존재할 수 있습니다.

그런 다음 추출된 각 종양 영역을 128 * 128 해상도로 겹치지 않는 정사각형 patch로 자르고 공백 비율이 0.3보다 큰 patch는 필터링합니다.

마지막으로 각 WSI에 대해 무작위로 샘플링된 10개(N)의 patch로 가방을 구성하고, 가방에 그룹화할 수 없는 남은 patch는 폐기합니다.

실험에 사용된 23가지 임상 특성은 나이(수치), 종양 크기(수치), ER(범주형), PR(범주형), HER2(범주형)등 이며, 이는 clinical_info.xlsx 데이터셋에서 확인할 수 있습니다.

## [모델 테스트]

위에서 언급했듯이 WSI는 여러 개의 가방으로 나뉘며, 각 가방은 예측 확률을 얻기 위해 MIL 모델에 입력됩니다. 따라서 테스트 중에 WSI의 종합적인 예측 결과를 얻기 위해 모든 백의 평균 예측 확률을 계산하여 "결과 병합"을 수행합니다.
<div align="center">
    <img src="imgs/b.png" alt="c"/>
</div>


## [평가 방법]

- Macro F1 Score로 결과를 평가합니다.
- Early Stopping F1 Score는 0.95로 합니다. 

---

## [Setup]

### Clone this repo

```bash
$ git clone https://github.com/zivary/DACON_AI_competition_for_predicting_Lymph_node_metastasis_of_breast_cancer.git
```

### Environment

Create environment and install dependencies.

```bash
$ conda create -n env python=3.8 -y
$ conda activate env
$ pip install -r ./code/requirements.txt
```

### Data Preparation

```text
code/Data_Preparation.ipynb
```

### MIL Modeling

```text
code/MIL.ipynb
```

## [결과]

결과적으로 공유된 Baseline의 결과 수준의 F1 스코어에 근접한 수준의 모델을 만드는 것에 만족해야 했습니다.

Crop 이미지의 크기조정, 배경 노이즈 제거, 이미지 증식,  다양한 학습모델 사용, 서로 다른 형식의 MIL 적용 등 여러 가지 방식의 시도를 해보았지만 뚜렷하게 학습 결과를 향상하는 방법을 찾지 못했습니다.

만족할 만한 결과를 얻지 못했지만, Git-Hub에 공개된 여러 논문들 이해하여 우리의 데이터에 맞춰 적용해 보고, 딥러닝 모델을 학습시켜 원하는 방식의 답을 얻어보는 뜻깊은 프로젝트였습니다. 또 프로젝트 과정에서 접해보지 못했던 다양한 Python 라이브러리들을 경험해 볼 수 있었습니다. 

---

## [참조]
이 프로젝트는 해당 오픈소스 프로젝트 기반으로 작성했습니다. 소스 코드를 공개해 주신 작성자에게 감사드립니다.
- https://github.com/bupt-ai-cz/BALNMP
- https://github.com/AMLab-Amsterdam/AttentionDeepMIL