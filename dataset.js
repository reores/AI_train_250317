
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
d2.set_img(0,{imgtitle:"",imglog:"",imgurl:"",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
let d2 = new DataSet("분류모델 구현")//메인타이틀
d2.set_content("패션 mnist 회귀 다중 분류")//서브 타이틀
d2.set_img(0,{imgtitle:"1. 패션데이터 수신",imglog:"",imgurl:"https://drive.google.com/file/d/1ZAOfCO7teED1_eW8KnBNzBz_bxCyWsZf/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"2. 데이터 구조 확인",imglog:"",imgurl:"https://drive.google.com/file/d/1N6cXzqTwFgV-O01iRfMFP8oYkOM1codH/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"3. 데이터 정규하 및 확인",imglog:"",imgurl:"https://drive.google.com/file/d/1rqCWd-dKqRfapRTlWSCLRXFBqsn88MtN/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"4. 데이터 셔플링 및 정답 일치성 확인",imglog:"",imgurl:"https://drive.google.com/file/d/17dHWDAPKsnVKZ1Cu9N0oaQa2tDiuWwyp/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"5. 커스텀 원핫인코딩 수행",imglog:"",imgurl:"https://drive.google.com/file/d/1uDvaE1fhJN4JosIz9lSNqeTe6GF0luSN/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"6. 모델구성 및 훈련실행",imglog:"",imgurl:"https://drive.google.com/file/d/1n4rNzKxVgE-nG0oz5f7ffoRtg_K9gDtn/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"7. 훈련결과 시각화",imglog:"",imgurl:"https://drive.google.com/file/d/1qjJlNEEqagNVmTfzMN2AGB0hyhtcDLhA/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"8. 예측결과 시각화",imglog:"",imgurl:"https://drive.google.com/file/d/11sU6P_j8dPB_1Dtprbfwi0shvNxUOOnI/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"9. 모델 및 인코더 저장",imglog:"",imgurl:"https://drive.google.com/file/d/1xu0LzQvxiaoYLa8dkUcLdh6CjJvpYBU_/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"10. 모델 및 인코더 불러오기",imglog:"",imgurl:"https://drive.google.com/file/d/1ixyiHTPmlIK3L-6pJe_pNxzj3xpMqiQd/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"11. 파일불러오기 및 배경제거_정규화",imglog:"",imgurl:"https://drive.google.com/file/d/1MtLA4UGYphLb1pjEVyDYpZJ6i5ZJ0kBv/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"12. 실제 이미지 결과 시각화",imglog:"",imgurl:"https://drive.google.com/file/d/1Tu-0tjycj9Bp0SRzgDMY40CtWAyiJR0Z/view?usp=drive_link",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/ClassificationSoftmax_fashionMnist/Examp_classification_fashionMnist.py"})//이미지타이틀

d1.set_fill("회귀 모델의 softmax 다중 분류 구현")//사용자 에필로그
data_sets.push(d2)

// menu3 =============================================================
let d3 = new DataSet("서버프로그램구현")//메인타이틀

data_sets.push(d3)

// menu4 =============================================================
let d4 = new DataSet("배치프로그램구현")//메인타이틀

data_sets.push(d4)
