
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

//d1.set_img(0,{imgtitle:"",imglog:"",imgurl:"",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_bostonHousing.py"})//이미지타이틀
let d1 = new DataSet("선형회귀모델")//메인 타이틀 //메뉴번호
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
d1.set_img(1,{imgtitle:"캘리포니아 주택가격 예측",imgurl:"",imglog:"",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_bostonHousing.py"})//이미지타이틀

d1.set_content("당뇨상태 1년 후 예측 선형 회귀모델")
d1.set_img(2,{imgtitle:"1년 후 당뇨상태 예측",imgurl:"",imglog:"",sourceurl:"https://github.com/reores/AI_train_250317/blob/main/LinearRegression/Examp_LinearRegression_bostonHousing.py"})//이미지타이틀

d1.set_fill("선형 회귀모델은 단일 데이터 또는 다중 데이터를 이용하여 연속적인 값을 출력하여 예측한다.")//사용자 에필로그
data_sets.push(d1)

// menu2 =============================================================
let d2 = new DataSet("공통모듈구현")//메인타이틀

data_sets.push(d2)

// menu3 =============================================================
let d3 = new DataSet("서버프로그램구현")//메인타이틀

data_sets.push(d3)

// menu4 =============================================================
let d4 = new DataSet("배치프로그램구현")//메인타이틀

data_sets.push(d4)
