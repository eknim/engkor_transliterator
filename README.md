# Eng-Kor Transliteration

하나의 언어에서 다른 언어로 변환할 수 있는 방식이 2가지가 있습니다. 첫 번째, 의미를 이해하고 이를 토대로 변환하는 translation이 있습니다. 예를 들어, 'Hello'라는 영어 인사말을 '안녕'이라는 한국어 인사말로 변환할 수 있습니다. 두 번째, 의미를 이해하진 않지만 소리를 토대로만 변환하는 transliteration이 있습니다. 예를 들어, 영어 'Hello'를 한글 '헬로우'로 소리나는대로 변환할 수 있습니다. 물론, translation이 훨씬 더 활용도가 강하지만, 저는 검색팀에 속해 브랜드 자동 매핑 업무를 수행하면서 transliteration을 부분적으로 활용할 수 있었습니다. '나이키', '아디다스'와 같이 외국 브랜드를 한국어로 명명할 때 보통 소리나는대로 명시하기에 영문 브랜드명을 한글로 변환할 필요가 있었습니다.

일반적인 자연어처리 문제와 같이 transliteration도 문맥에 따라 똑같은 문자셋이라도 발음이 달라지는 경향을 가집니다. 이에 규칙기반의 프로세스를 선택하기 보다는 통계기반의 기계학습(딥러닝) 모델을 구축하는 것이 더 현명합니다. 다양한 딥러닝 모델 중에서 '다:다' 매핑을 할 수 있는 Sequence to sequence (Seq2seq) 모델을 선택했고, 최근 자연어처리의 필수로 자리잡히고 있는 attention 메커니즘도 같이 사용했습니다. Encoder로부터 생성되는 context 벡터를 attention 메커니즘으로부터 더 정교하게 모델링할 수 있기에 예측을 좀 더 정확히 할 수 있습니다. 참고로 Seq2seq 모델은 translation 문제를 해결하기 위해 개발된 모델입니다. 

다음은 본 프로젝트에 활용한 Seq2seq with attention 모델을 표현한 그림입니다. (학습이 완료된 모델임을 가정) "attention"이 입력되면 "어텐션"이 출력됨을 나타내고 있습니다. Encoder와 Decoder는 LSTM(Long Short-Term Memory)으로, attention은 기본적인 NN(Neural Network) 모듈로 구성됩니다. NN가 있다는 말은 학습이 가능한 가중치가 있다는 말과 같습니다. 즉, 입력 단위로 세기를 조절하여 중요도를 구분할 수 있습니다. 예를 들어, 'a'와 't'의 신호를 증폭시켜 'ㅓ'를 출력할 수 있도록 유도합니다.

![](/assets/seq2seq_att_diagram.png)

영문과 같이 한글의 vocabulary 구성도 음소단위(ex.자음,모음)로 구성했습니다. 이를 통해 vocab 사이즈와 학습해야 되는 패턴을 최소화할 수 있고, 높은 일반성을 가지기에 학습 시그널이 부족한 단어가 들어와도 잘 대응할 수 있다는 기대를 가질 수 있습니다. 다음 표를 통해 음절단위로 구성된 muik 모델과 음소단위로 구성된 본 프로젝트의 seq2seq_att 모델과의 결과를 비교할 수 있습니다. 두 모델은 똑같은 데이터셋으로 학습하였습니다. 다음 예제는 모두 학습 데이터 존재하지 않는 단어들입니다.

| input           | muik          |    seq2seq_att    |
| :-------------: |:-------------:|:------:|
| attention      | 애텐션           |   어텐션   |
| tokenizer      | 토케니저         |   토크나이저   |
| transliterator | 트랜슬리테라토르   | 트랜슬리터레이터 |
| suddenly       | 서들리          | 서든리
| mecab          | 메카            | 메카브   |

(참고) gritmind처럼 두 개의 단어(grit, mind)가 하나로 합쳐진 단어일 경우, 언더바를 중간에 추가하고 입력에 넣을 경우 결과가 더 잘 나올 수 있습니다. 예) 'gritmind'->'그리트민드', 'grit_mind'->그릿마인드


## Prerequisites
* tensorflow 1.13.1
* keras 2.2.4
* numpy 1.16.1
* hgtk 0.1.3
* matplotlib 3.0.3


## Usage

### 1. Train model

```python
from engkor_transliterator import seq2seq_att
model = seq2seq_att.Transliterator()
model.train() # train
model.decode_sequence('attention') # input: attention
>>>
('어텐션', 0.9819696)
```

### 2. Use pre-trained model

```python
from engkor_transliterator import seq2seq_att
model = seq2seq_att.Transliterator()
model.use_pretrained_model() # use pre-trained model
model.decode_sequence('attention') # input: attention
>>>
('어텐션', 0.9819696)
```


## References

데이터셋은 muik의 [transliteration](https://github.com/muik/transliteration) 레파지토리에 있는 데이터를 그대로 사용하였고, Seq2seq attention 모델의 소스코드 부분은 lazyprogrammer의 udemy, [deep-learning-advanced-nlp](https://deeplearningcourses.com/c/deep-learning-advanced-nlp
) 강의를 참조하였습니다. muik의 모델과의 비교했을 때, 똑같이 sequence to sequence 모델을 사용했지만, attention을 추가적으로 사용했고, 한글을 음절단위가 아닌 음소단위로 구분하였습니다.




