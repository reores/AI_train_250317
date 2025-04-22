
//메뉴 생성기 종료 E==============================
//데이터 아키텍처{sub_title:"",sub_content:"",sub_img:[],user_fill:""}
let data_sets=[]
class DataSet{
	constructor(sub_title,menuNum){this.sub_title=sub_title}
	user_fill=""
	sub_content=[]
	sub_img=[]
	set_content(content){this.sub_content.push(content)}
	set_img(num,obj){
		if(!this.sub_img[num]){this.sub_img[num]=[]}
		this.sub_img[num].push(obj)
	}
	set_fill(ufill){this.user_fill=ufill}	 
}

// menu2 =============================================================
//d1.set_img(0,{imgtitle:"",imglog:"",imgurl:"",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})//이미지타이틀
let d1 = new DataSet("선형 회귀모델")//메인 타이틀 //메뉴번호
d1.set_content("보스턴 주택가격 예측 선형 회귀모델")//서브 타이틀
d1.set_img(0,{imgtitle:"1. 보스턴 데이터 수신",imglog:"텐서플로우 보스턴 데이터셋 수신코드",imgurl:"https://drive.google.com/file/d/1cCr9IHyF1SWtMT1fzXZgEzip7RNLg9US/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"2. 데이터 특성 파악",imglog:"각 필드별 데이터의 특성과 의미 및 값을 확인",imgurl:"https://drive.google.com/file/d/1I7I3EF73NFjj2XN_GULx-4GHl9aOo1r1/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"3. 데이터 연관성 확인(산점도)",imglog:"가격과 데이터의 특성별 상호 연관성을 파악",imgurl:"https://drive.google.com/file/d/1ojQ4Dj6_dDMvxsPbruO5rq1hvJ5pVQF-/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"4. 데이터 분포도 확인(히스토그램)",imglog:"히스토그램을 이용하여 데이터의 분포와 이상치 데이터 확인",imgurl:"https://drive.google.com/file/d/1f8fKCVBLhClc9wT1WhZbTrHhNMCrUAb4/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"5. 데이터 정규분포 전환",imglog:"훈련 전처리를 위해 연관성이 있는 데이터 가공 : 평균 0, 표준편차 1로 구성된 정규분포로 변환",imgurl:"https://drive.google.com/file/d/18hR73K6KGBpAGS_7z0_susqaPnEnpQJ8/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"6. 순차모델 구성 및 훈련실행",imglog:"평균제곱오차법(MSE)을 이용한 손실함수와 경사하강법(SGD)을 이용한 최적화함수로 컴파일 및 훈련 15회 실행",imgurl:"https://drive.google.com/file/d/1CNG64EZlQQAs762pPYsG2VMme3zGsjTf/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"7. 훈련결과 시각화",imglog:"훈련 결과인 MSE 손실값의 변화를 시각화 표현",imgurl:"https://drive.google.com/file/d/1iv3TqbfWt5x_mnng2qnkL_6a-ZRq1XQ7/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"8. 모델 구동 테스트",imglog:"테스트 데이터를 주입하여 예측결과를 인출하고 실제 정답과 차이를 정확률로 표기",imgurl:"https://drive.google.com/file/d/1N1DwX2HwYZNi6rEFf_Hr_Dz3OkoGgPP0/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_bostonHousing.py"})//이미지타이틀

d1.set_content("캘리포니아 주택가격 예측 선형 회귀모델")
d1.set_img(1,{imgtitle:"1. 캘리포니아 주택 특성데이터 수신 및 분석",imglog:"사이킷런에서 제공하는 캘리포니아 주택 가격에 따른 데이터 특성(x)들의 모음과 가격정보(y)",imgurl:"https://drive.google.com/file/d/19I52Z00gXR6t1cgh0lz7M6GtBU33hixT/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})//이미지타이틀
d1.set_img(1,{imgtitle:"2. 주택특성과 가격의 연관성 분석",imglog:"주택의 특성별 산점도 분석으로 가격에 따른 선형성 확인",imgurl:"https://drive.google.com/file/d/1pYJYgfRsDXAtVJ4ArhGuOjM_cILhGEYL/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})//이미지타이틀
d1.set_img(1,{imgtitle:"3. 데이터 통계정보 분석",imglog:"판다스 DataFrame으로 변환 후 평균치, 표준편차 등의 데이터 통계정보 분석",imgurl:"https://drive.google.com/file/d/1YqBxWwzDgJIjB1Fx4512hRPERSuUBURm/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})//이미지타이틀
d1.set_img(1,{imgtitle:"4. 데이터 분포 확인",imglog:"히스토그램으로 데이터 분포 시각화와 이상데이터 또는 범위를 벗어난 데이터 설정(임계치 산정)",imgurl:"https://drive.google.com/file/d/1JnuH1GhnD4fJmvygotm_VINyJ_wA9TWN/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})//이미지타이틀
d1.set_img(1,{imgtitle:"5. 이상 데이터 제거",imglog:"범위를 벗어나거나 이상치 데이터는 성능에 치명적인 영향을 줄 수 있으므로 제거하여 데이터 정제를 수행",imgurl:"https://drive.google.com/file/d/1e63z-snTZD1j_quwkFDesSsWEVT8EzCZ/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})//이미지타이틀
d1.set_img(1,{imgtitle:"6. 데이터 정제 후 분포확인",imglog:"데이터 전처리 시행 후 데이터 범위 등을 히스토그램을 통해 이상데이터 분포 확인",imgurl:"https://drive.google.com/file/d/1NUDJ2FaptTQvyFdMxD8wy_tbVQTXpgMK/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})//이미지타이틀
d1.set_img(1,{imgtitle:"7. 훈련데이터와 테스트 데이터 분할",imglog:"사이킷런 라이브러리를 이용해 훈련데이터 80%, 테스트데이터 20% 비율로 분할 및 데이터 정규분포화 실행",imgurl:"https://drive.google.com/file/d/1e_45dq8bMKG3CRc8KCla8RcvTVU7dXgf/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})//이미지타이틀
d1.set_img(1,{imgtitle:"8. 선형회귀 기계학습 모델 구성과 훈련 ",imglog:"은닉층이 존재하지 않는 머신러닝 모델을 구성하고 평균제곱오차 손실함수와 경사하강법 최적화 함수를 설정한 후 훈련 100회 실행",imgurl:"https://drive.google.com/file/d/1OqFpMkQtzgOSDGRbOyoLynEy123Fk_RM/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})//이미지타이틀
d1.set_img(1,{imgtitle:"9. 훈련결과 시각화",imglog:"훈련시 저장된 손실값을 이용하여 시각화 그래프 표현",imgurl:"https://drive.google.com/file/d/1g6WEe8nHqRoN0tH0xGV_7839N0zVP8Bj/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})//이미지타이틀
d1.set_img(1,{imgtitle:"10. 정확률 계산해보기",imglog:"테스트 데이터를 측정 후 실제 정답과 예측 값을 비교해 정확률 판단",imgurl:"https://drive.google.com/file/d/14Zxu_7Q6UZkLImvUuqR7f1zZvYNMqSHT/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})//이미지타이틀

d1.set_content("당뇨상태 1년 후 예측 선형 회귀모델")
d1.set_img(2,{imgtitle:"1년 후 당뇨상태 예측",imglog:"",imgurl:"",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_bostonHousing.py"})//이미지타이틀

d1.set_fill("선형 회귀모델은 단일 데이터 또는 다중 데이터를 이용하여 연속적인 값을 출력하여 예측한다.")//사용자 에필로그
data_sets.push(d1)

// menu2 =============================================================
//d2.set_img(1,{imgtitle:"",imglog:"",imgurl:"",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classification_Convolution/Exam_fashionmnist_class_conv.py"})//이미지타이틀
let d2 = new DataSet("분류모델 구현")//메인타이틀
d2.set_content("패션 mnist 회귀 다중 분류")//서브 타이틀
d2.set_img(0,{imgtitle:"1. fashion_mnist 데이터 수신",imglog:"구글에서 제공하는 패션관련 이미지 다운로드",imgurl:"https://drive.google.com/file/d/1ZAOfCO7teED1_eW8KnBNzBz_bxCyWsZf/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"2. 수신 데이터 구조 확인",imglog:"훈련데이터 6만개, 테스트 데이터 1만개, 이미지 사이즈 28x28, 1채널 Gray, 정답데이터 정수형",imgurl:"https://drive.google.com/file/d/1N6cXzqTwFgV-O01iRfMFP8oYkOM1codH/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"3. 훈련데이터 minmax 정규화",imglog:"훈련데이터의 0 ~ 1 사이 값으로 정규화 실행",imgurl:"https://drive.google.com/file/d/1rqCWd-dKqRfapRTlWSCLRXFBqsn88MtN/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"4. 데이터 셔플링 및 정답 일치성 확인",imglog:"데이터 셔플링 후 정답과 일치하도록 셔플이 되었는지 확인",imgurl:"https://drive.google.com/file/d/17dHWDAPKsnVKZ1Cu9N0oaQa2tDiuWwyp/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"5. 커스텀 원핫인코딩 수행",imglog:"커스텀 원핫인코딩 클래스 생성 후 원하는 방법으로 원핫인코딩 후 작동 확인",imgurl:"https://drive.google.com/file/d/1uDvaE1fhJN4JosIz9lSNqeTe6GF0luSN/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/custom_encoder.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"6. 모델 컴파일 및 훈련실행",imglog:"flatten으로 완전층 연결 후 다중분류 모델로 softmax 활성화 함수 사용하여 10개의 class 출력",imgurl:"https://drive.google.com/file/d/1n4rNzKxVgE-nG0oz5f7ffoRtg_K9gDtn/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"7. 훈련결과 시각화",imglog:"훈련 종료 후 손실 및 정확도 시각화 표현",imgurl:"https://drive.google.com/file/d/1qjJlNEEqagNVmTfzMN2AGB0hyhtcDLhA/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"8. 테스트데이터 예측",imglog:"모델의 테스트 데이터 예측과 예측 결과 시각화",imgurl:"https://drive.google.com/file/d/11sU6P_j8dPB_1Dtprbfwi0shvNxUOOnI/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"9. 모델 및 인코더 저장",imglog:"훈련된 모델의 저장과 레이블이 연결된 인코더 파일로 저장",imgurl:"https://drive.google.com/file/d/1xu0LzQvxiaoYLa8dkUcLdh6CjJvpYBU_/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"10. 모델 및 인코더 불러오기",imglog:"실제 이미지 측정을 위해 훈련된 모델과 라벨 인코더를 호출",imgurl:"https://drive.google.com/file/d/1ixyiHTPmlIK3L-6pJe_pNxzj3xpMqiQd/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"11. 실제 이미지 테스트",imglog:"인터넷의 이미지를 복사하여(test_img) 모델에 적합하도록 전처리 후 테스트 실행",imgurl:"https://drive.google.com/file/d/1MtLA4UGYphLb1pjEVyDYpZJ6i5ZJ0kBv/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"12. 실제 이미지 예측결과 시각화",imglog:"실제 이미지를 예측한 결과를 시각화 하여 표현",imgurl:"https://drive.google.com/file/d/1Tu-0tjycj9Bp0SRzgDMY40CtWAyiJR0Z/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀

d2.set_content("패션 mnist CNN 모델")//서브 타이틀
d2.set_img(1,{imgtitle:"1. 데이터 불러오기 및 라벨리스트 생성",imglog:"텐서플로우에서 테스트용 패션 mnist 데이터를 불러온다.",imgurl:"https://drive.google.com/file/d/1tHe5YXNq-y_p4f-okb_TUGugs_kr_r8A/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classification_Convolution/Exam_fashionmnist_class_conv.py"})//이미지타이틀
d2.set_img(1,{imgtitle:"2. 데이터 구조확인",imglog:"불러온 데이터 내부 구조를 확인한다.",imgurl:"https://drive.google.com/file/d/1SieUzgMifHbkVM1JVNsThCsXYRqmf0Gv/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classification_Convolution/Exam_fashionmnist_class_conv.py"})//이미지타이틀
d2.set_img(1,{imgtitle:"3. 데이터 분할",imglog:"사이킷런의 train_test_split 을 활용해 검증 데이터를 분리한다.",imgurl:"https://drive.google.com/file/d/19IS9fc-SVqGjhCKYKNL8rpigNOiUj5IX/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classification_Convolution/Exam_fashionmnist_class_conv.py"})//이미지타이틀
d2.set_img(1,{imgtitle:"4. 데이터 셔플 및 전처리",imglog:"사이킷런의 shuffle 을 활용해 데이터를 섞어준다.",imgurl:"https://drive.google.com/file/d/1YFxeYO2XHnx6Dq4nVoINnUZSDpE9YsA2/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classification_Convolution/Exam_fashionmnist_class_conv.py"})//이미지타이틀
d2.set_img(1,{imgtitle:"5. 정답 및 이미지 일치여부 확인",imglog:"임의의 데이터를 뽑아 정답과 이미지 일치 여부를 확인한다.",imgurl:"https://drive.google.com/file/d/1U51eVypQEky72_N2Gu1JfaSh4-Ur9TCw/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classification_Convolution/Exam_fashionmnist_class_conv.py"})//이미지타이틀
d2.set_img(1,{imgtitle:"6. 모델구성 및 컴파일",imglog:"Convolution Layer(Conv2D)가 적용된 모델을 구성한다.",imgurl:"https://drive.google.com/file/d/1iee-cq20PlzNjncuijUMOMs9fc51rSLF/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classification_Convolution/Exam_fashionmnist_class_conv.py"})//이미지타이틀
d2.set_img(1,{imgtitle:"7. 훈련실행",imglog:"데이터 개수를 고려하여 배치사이즈 약배수로 지정 후 훈련 100회 실행",imgurl:"https://drive.google.com/file/d/1DeSFR3VGLqEzD11pW0YvG21LpATPJk3r/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classification_Convolution/Exam_fashionmnist_class_conv.py"})//이미지타이틀
d2.set_img(1,{imgtitle:"8. 그래프그리기",imglog:"훈련 정확도와 손실율을 그래프로 그려 시각화",imgurl:"https://drive.google.com/file/d/1qmbUsUHPZNz-lVwefE_n6oa6jpUrUx-r/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classification_Convolution/Exam_fashionmnist_class_conv.py"})//이미지타이틀
d2.set_img(1,{imgtitle:"9. 모델평가 및 예측 시각화",imglog:"모델평가(evaluate), 예측(predict) 후 예측결과 시각화",imgurl:"https://drive.google.com/file/d/1gyRDbDr7sH8i22j-V9XUad776LBbFSU2/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classification_Convolution/Exam_fashionmnist_class_conv.py"})//이미지타이틀
d2.set_img(1,{imgtitle:"10. 혼동행렬 생성",imglog:"예측값과 실제 정답의 데이터 구조를 일치화시키고 confusion_metrix로 혼동행렬 생성",imgurl:"https://drive.google.com/file/d/1dY8npU47jGmViUNBXyhjFLFwMZMZMWyU/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classification_Convolution/Exam_fashionmnist_class_conv.py"})//이미지타이틀
d2.set_img(1,{imgtitle:"11. 혼동행렬 시각화(히트맵)",imglog:"생성된 혼동행렬을 seaborn의 heatmap을 활용해 시각화",imgurl:"https://drive.google.com/file/d/1Ktp3N4RnTeBAEvZVnPLDMyItDF1UGZbs/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classification_Convolution/Exam_fashionmnist_class_conv.py"})//이미지타이틀
d2.set_img(1,{imgtitle:"12. f1스코어 및 모델저장",imglog:"f1 스코어를 확인 후 모델 최종 저장",imgurl:"https://drive.google.com/file/d/1FDo5G20BMp2HmbOA7t2hZw8uaHcsQLjf/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classification_Convolution/Exam_fashionmnist_class_conv.py"})//이미지타이틀

d2.set_content("암호화폐 회귀분석 가격예측")//서브 타이틀
d2.set_img(2,{imgtitle:"1. 가상화폐 데이터수신 모듈",imglog:"날짜별/주별/시간별/화폐별 데이터 수신 모듈",imgurl:"https://drive.google.com/file/d/1hC9fw7hHFQw6ZyR1WDTMKB0oa_oDljNM/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Encrypto_Anal/utilpy.py"})//이미지타이틀
d2.set_img(2,{imgtitle:"2. 데이터 셋 생성하기",imglog:"가격정보와 연관성이 있는 값으로 구성된 데이터 셋",imgurl:"https://drive.google.com/file/d/1QNaACharG6UqrLeVsgIScUjE9Ii-VSVP/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Encrypto_Anal/utilpy.py"})//이미지타이틀
d2.set_img(2,{imgtitle:"3. 데이터 값 추출하기",imglog:"문제데이터와 필드리스트 추출 모듈",imgurl:"https://drive.google.com/file/d/1z1djho-kujseirmkb0Ic1Pdl5-cqQGEX/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Encrypto_Anal/utilpy.py"})//이미지타이틀
d2.set_img(2,{imgtitle:"4. 모듈을 통해 데이터 수신",imglog:"데이터 수신 후 생성/추출 모듈 호출",imgurl:"https://drive.google.com/file/d/1NWzn0g_v7HYqYzuH-NDjOuG3BkAcbEg9/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Encrypto_Anal/EncryptoCoinPredPrice.py"})//이미지타이틀
d2.set_img(2,{imgtitle:"5. 산점도 연관성 분석",imglog:"산점도 연관성 분석으로 가격정보와 연관성이 낮은 필드 제거",imgurl:"https://drive.google.com/file/d/1VxKzb5nJtD42fXMMIYJQYIpnkCsS7dOF/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Encrypto_Anal/EncryptoCoinPredPrice.py"})//이미지타이틀
d2.set_img(2,{imgtitle:"6. 회귀분석을 위한 파생 값 통합",imglog:"회귀분석을 위해 연관 필드 데이터 통합 및 정규화",imgurl:"https://drive.google.com/file/d/1ceyskd-unMjzTM4pefQvHp6P91CvHYzP/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Encrypto_Anal/EncryptoCoinPredPrice.py"})//이미지타이틀
d2.set_img(2,{imgtitle:"7. 심층모델 구성",imglog:"다층 레이어로 구성된 회귀모델 구성 및 컴파일 / 훈련",imgurl:"https://drive.google.com/file/d/1VmNSBpTBf7WJkxZdt9eVbt_0jX3-OYx_/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Encrypto_Anal/EncryptoCoinPredPrice.py"})//이미지타이틀
d2.set_img(2,{imgtitle:"8. 예측 및 확률 검증",imglog:"예측은 산점도, 실제는 선형그래프로 시각화 비교",imgurl:"https://drive.google.com/file/d/15ExMsFgJT44mX8jF3EbjUcz_dH5zZDvT/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Encrypto_Anal/EncryptoCoinPredPrice.py"})//이미지타이틀
d2.set_img(2,{imgtitle:"9. 예측정보 출력",imglog:"다음 사이클(내일)의 가격 예측",imgurl:"https://drive.google.com/file/d/1ycMY5e0tQKXQGehN__3gpWsHj7Vtp1jc/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Encrypto_Anal/EncryptoCoinPredPrice.py"})//이미지타이틀
d2.set_img(2,{imgtitle:"10. 예측과 실제 그래프 시각화",imglog:"예측정보와 실제그래프를 동시 표현으로 정확도 판단",imgurl:"https://drive.google.com/file/d/1o1DqpycXF6mrYSQ3DY5wQ--O4a8rNrUw/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Encrypto_Anal/EncryptoCoinPredPrice.py"})//이미지타이틀

d2.set_content("mnist 데이터 컨볼루션 다중분류")//서브 타이틀
d2.set_img(3,{imgtitle:"1. 데이터 수신 및 셔플",imglog:"mnist 손으로 쓴 숫자데이터 수신",imgurl:"https://drive.google.com/file/d/1VCBhp4eN3Z7aY8GwmyILPmN15w2q4uqp/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classifi_mnist_conv_advan/mnist_conv_main.py"})//이미지타이틀
d2.set_img(3,{imgtitle:"2. 데이터 스케일 및 원핫인코딩",imglog:"min-max scaling 정답데이터 원핫인코딩",imgurl:"https://drive.google.com/file/d/1zq1v0Q8AFotZHTpAxjoHrPS6k-rCzyqh/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classifi_mnist_conv_advan/mnist_conv_main.py"})//이미지타이틀
d2.set_img(3,{imgtitle:"3. 콜백함수 생성",imglog:"체크포인트 모델 저장 콜백과 훈련 조기종료 콜백 정의",imgurl:"https://drive.google.com/file/d/1XEOIIuKX3uKki9Ny8uaE4Qwlu27hzCR6/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classifi_mnist_conv_advan/mnist_conv_main.py"})//이미지타이틀
d2.set_img(3,{imgtitle:"4. 모델 컴파일 및 훈련",imglog:"컨볼루션 레이어 적용한 모델 구성 및 훈련",imgurl:"https://drive.google.com/file/d/1bPCwMucv0kmWH-qA-HRVCo5XBzKuBFPA/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classifi_mnist_conv_advan/mnist_conv_main.py"})//이미지타이틀
d2.set_img(3,{imgtitle:"5. 모델 조기종료 및 최적값 체크포인트",imglog:"모델 저장/조기종료 시점 및 최적값 확인",imgurl:"https://drive.google.com/file/d/1cn5G0wOG0xGyGAJwGhfmR1RnNn22YCLA/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classifi_mnist_conv_advan/mnist_conv_main.py"})//이미지타이틀
d2.set_img(3,{imgtitle:"6. 훈련시각화 및 모델 선택",imglog:"훈련결과 시각화를 통해 최적화된 epochs 판단",imgurl:"https://drive.google.com/file/d/14ejsIk9K35FB_W7wahe7mQ-eppbLaAzQ/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classifi_mnist_conv_advan/mnist_conv_main.py"})//이미지타이틀
d2.set_img(3,{imgtitle:"7. 모델 선택",imglog:"최적화된 epochs에 해당하는 모델 호출",imgurl:"https://drive.google.com/file/d/1nYP1X9s3HXKxD-ewWAxXabXOZQTx3K8w/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classifi_mnist_conv_advan/mnist_conv_main.py"})//이미지타이틀
d2.set_img(3,{imgtitle:"8. 테스트 데이터 예측 결과",imglog:"불러온 모델을 통해 테스트 데이터 예측해보기",imgurl:"https://drive.google.com/file/d/1Z_BDJ88gOO2GFpWtmPSqs_1iYHscqaaj/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classifi_mnist_conv_advan/mnist_conv_main.py"})//이미지타이틀
d2.set_img(3,{imgtitle:"9. 그림판 이미지 판별 테스트",imglog:"직접 그린 그림판 이미지 판별해보기",imgurl:"https://drive.google.com/file/d/1RGdLDjHsGUlsK5h_eFfJ5472iqL-lKA1/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classifi_mnist_conv_advan/Test_mnist_conv_advan.py"})//이미지타이틀
d2.set_img(3,{imgtitle:"10. 그림판 이미지 예측 결과",imglog:"직접 그린 그림판 이미지 판결 결과",imgurl:"https://drive.google.com/file/d/1bZtK1w8JzqlUskQJXq8p5Nx8PxAvF3BN/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/Classifi_mnist_conv_advan/Test_mnist_conv_advan.py"})//이미지타이틀
d2.set_img(3,{imgtitle:"11. 자동 저장 모델",imglog:"로컬 지정경로에 저장된 자동 저장 모델 현황",imgurl:"https://drive.google.com/file/d/1UkCkTgt5bdckELCQPmOSI4tPnVghRBMF/view?usp=drive_link",sourceurl:""})//이미지타이틀

d2.set_fill("회귀 모델의 softmax 다중 분류 구현")//사용자 에필로그
data_sets.push(d2)

// menu3 =============================================================
let d3 = new DataSet("NLP_RNN 시계열 순환모델")//메인타이틀
d3.set_content("가상화폐 가격 분석")//서브 타이틀
d3.set_img(0,{imgtitle:"1. 데이터 API 수신",imglog:"빗썸 캔들데이터 날짜별 수신 모듈 ",imgurl:"https://drive.google.com/file/d/1d6kK-gOmweHghQqOQjDXbuTFWb9nIMnK/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/tree/main/RNN_encrypto_coin"})//이미지타이틀
d3.set_img(0,{imgtitle:"2. 훈련데이터 생성",imglog:"훈련 데이터 정제 및 정답 데이터 생성 모듈",imgurl:"https://drive.google.com/file/d/1CVZUuvBnM_2F_Gjm8-_niC8lZ0H6HLGY/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/tree/main/RNN_encrypto_coin"})//이미지타이틀
d3.set_img(0,{imgtitle:"3. 예측데이터 생성",imglog:"오늘 및 전일 예측 데이터 생성 모듈",imgurl:"https://drive.google.com/file/d/1IUI4rYNrvupwwVKW7wn3ouRweiG1Sc9D/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/tree/main/RNN_encrypto_coin"})//이미지타이틀
d3.set_img(0,{imgtitle:"4. 데이터 상관관계 분석 스케터",imglog:"수신된 필드별 데이터의 상관관계 분석을 위한 스케터",imgurl:"https://drive.google.com/file/d/17WrylUfroxF3wltJjwuUYxjkP9n8XzOw/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/tree/main/RNN_encrypto_coin"})//이미지타이틀
d3.set_img(0,{imgtitle:"5. 스케일러 함수",imglog:"필드별 표준 정규분포 데이터 스케일링 평탄화",imgurl:"https://drive.google.com/file/d/1qakrynMBfjn9uqC4YeA3Xp0pyXN6gGbb/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/tree/main/RNN_encrypto_coin"})//이미지타이틀
d3.set_img(0,{imgtitle:"6. ConvLSTM 모델 구성",imglog:"ConvLSTM 모델과 양방향 학습으로 구성된 모델 구성 후 컴파일",imgurl:"https://drive.google.com/file/d/1dMrgxdsvC9bGIkdClNOYkxzSIKghMybs/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/tree/main/RNN_encrypto_coin"})//이미지타이틀
d3.set_img(0,{imgtitle:"7. 훈련실행 및 체크포인트 콜백함수 적용",imglog:"체크포인트 콜백함수로 validation loss 최적화 모델 선택",imgurl:"https://drive.google.com/file/d/1gCVZcGlejzPnYZQkg1HyIhTfvMRau_ls/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/tree/main/RNN_encrypto_coin"})//이미지타이틀
d3.set_img(0,{imgtitle:"8. 훈련결과 시각화",imglog:"MSE 데이터 결과 시각화 플롯그래프",imgurl:"https://drive.google.com/file/d/1jHdlL1G5RiIWB9vcnA_qH1y1Y13x30JO/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/tree/main/RNN_encrypto_coin"})//이미지타이틀
d3.set_img(0,{imgtitle:"9. 정답, 예측 산점도 시각화",imglog:"실제 정답과 예측 값 산점도 시각화 예측률 분석",imgurl:"https://drive.google.com/file/d/1j9qJ58dFPxyFtCnzNqstIBUREfhasjfI/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/tree/main/RNN_encrypto_coin"})//이미지타이틀
d3.set_img(0,{imgtitle:"10. 현재가격과 예측가격 산출을 위한 데이터 생성 및 예측",imglog:"현재가격과 현재가격산출 예측가격으로 오차율 산정 후 다음날의 예측가격 출력",imgurl:"https://drive.google.com/file/d/1rzAQLrbW93gdFtCGKZX2r9cmK82hgacR/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/tree/main/RNN_encrypto_coin"})//이미지타이틀
d3.set_img(0,{imgtitle:"11. 최고가 예측 LSTM 모델 구성",imglog:"LSTM 모델로 최고 가격 예측",imgurl:"https://drive.google.com/file/d/1Ej9L6XzD1mw_7_1zQnR6MiX5p5xG6aWi/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/tree/main/RNN_encrypto_coin"})//이미지타이틀
d3.set_img(0,{imgtitle:"12. 최고가 예측 산점도 및 예측가",imglog:"최고가 실제 가격과 예측 가격 산점도 비교 및 현재 가격의 예측 오차율과 다음 가격 산출",imgurl:"https://drive.google.com/file/d/1044vT_aYTVgxhnpxpIooogFA0-h-Cm54/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/tree/main/RNN_encrypto_coin"})//이미지타이틀

d3.set_fill("NLP_RNN 시계열 순환모델(선형회귀)")//사용자 에필로그
data_sets.push(d3)

// menu4 =============================================================
let d4 = new DataSet("배치프로그램구현")//메인타이틀

data_sets.push(d4)
