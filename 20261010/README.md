###### 전체 코드에서 과제가 아닌 부분은 모두 주석처리 했습니다.


<br />
<br />


#### $1)$ 데이터 형식 (Data Types)
문자열로 저장된 점수 "0.8825"를 실수로 변환한 후, 100을 곱하여 백분율로 표현했습니다.

이후 int() 함수를 사용해 정수로 변환했습니다.

score_str = "0.8825"

score_int = int(float(score_str) * 100)

print(f"최종 점수: {score_int}점")

실행 결과

최종 점수: 88점

개선점:
이전 코드는 다음과 같아서 타입 에러가 났었는데 int로 다시 형변환을 시켜주니 해결됐습니다.
score_int = float(score_str) * 100


<br />
<br />

#### $3)$ Enumerate & Zip

학생 이름과 점수가 저장된 리스트를 zip()으로 묶고, enumerate()를 이용해 학생 번호를 부여했습니다.

조건문을 사용하여 80점 이상인 학생의 정보만 출력했습니다.



for idx, student in enumerate(zip(students, scores), start=1):
    name, score = student
    if score >= 80:
        print(f"{idx}번 {name} 학생 (점수: {score}점)")

실행 결과

1번 짱구 학생 (점수: 85점)
2번 철수 학생 (점수: 92점)

개선점 : 
이전에는 for idx, name, score 변수를 3개를 선언해서 에러가 났었는데 변수를 개선해서 해결했습니다.

<br />
<br />

#### $5)$ NumPy


NumPy 배열에 대해 정규화를 한다는 것이 보통 Min-Max 정규화를 적용한다는 것을 검색해서 찾아 본 뒤에 코드로 작성했습니다.

import numpy as np

np_arr = np.array([1, 2, 3, 4, 5])

np_arr_normalized = (np_arr - np_arr.min()) / (np_arr.max() - np_arr.min())

print(np_arr_normalized)

실행 결과


-2

[0.   0.25 0.5  0.75 1.  ]

<br />
<br />

2.4 NumPy 배열을 float32로 변환

정규화한 배열을 astype() 함수를 사용해 float32 자료형으로 변환했습니다.

np_arr_final = ((np_arr - np_arr.min()) / (np_arr.max() - np_arr.min())).astype(np.float32)

print(np_arr_final)
print(np_arr_final.dtype)

<br />
<br />

##### 2. 학생 성적 분석 프로젝트

2.1 학생 이름 추출

List Comprehension과 split()을 활용하여 학번과 이름이 결합된 문자열에서 학생 이름만 추출했습니다.

student_names = [student.split("_")[1] for student in raw_students]

<br />
<br />

2.2 전체 성적 데이터 출력

flatten()을 사용하여 5명 학생의 3개 과목 성적을 1차원 배열로 변환했습니다.

all_scores = raw_scores.flatten()


<br />
<br />
2.3 85점 이상 성적 개수 계산

NumPy 조건 연산과 sum()을 활용하여 전체 15개 성적 중 85점 이상인 성적의 개수를 계산했습니다.

over_85 = (all_scores >= 85).sum()

<br />
<br />

2.4 학생별 평균 점수 계산

np.mean()과 axis=1을 이용하여 학생별 평균 점수를 계산했습니다.

student_means = np.mean(raw_scores, axis=1)


<br />
<br />
2.5 최고 평균 학생 선정

np.argmax()를 활용하여 평균 점수가 가장 높은 학생을 찾았습니다.

top_student_idx = np.argmax(student_means)
top_student_name = student_names[top_student_idx]

<br />
<br />

2.6 단일 과목 최고 점수 분석

NumPy의 max(), argmax()와 나머지 연산자 %, 정수 나눗셈 //를 활용하여 최고 점수의 학생과 과목을 계산했습니다.

overall_max_score = np.max(raw_scores)
overall_max_idx = np.argmax(raw_scores)
overall_max_subject = subjects[overall_max_idx % len(subjects)]
overall_max_student = student_names[overall_max_idx // len(subjects)]


<br />
<br />
#### 프로젝트 실행 결과

1. 학생 명단: 짱구, 철수, 맹구, 유리, 훈이

2. 전산 시스템에 입력된 모든 성적: 95점, 82점, 88점, 70점, 65점, 90점, 98점, 100점, 92점, 60점, 75점, 80점, 81점, 88점, 84점

3. 전체 15개 성적 중 85점 이상 개수: 7개

4. 학생별 평균 점수: 짱구 88.33점, 철수 75.00점, 맹구 96.67점, 유리 71.67점, 훈이 84.33점

5. 최고 평균 수석 학생: 맹구 (96.67점)

6. 단일 과목 최고 점수: 100점 (과목: 수학, 학생: 맹구)

<br />
<br />

#### 3. 실행 환경 및 의존성

사용 언어: Python 3

개발 환경: Visual Studio Code (VS Code)

제출 파일: 01_Linux-and-Python.py

사용 라이브러리: NumPy

필요한 라이브러리 설치

python -m pip install numpy

<br />
<br />

#### 4. 실행 방법

Python 3을 설치합니다.

Visual Studio Code에서 제출 파일이 있는 폴더를 엽니다.

터미널에서 NumPy 라이브러리를 설치합니다.

01_Linux-and-Python.py 파일을 엽니다.

해당 파일이 있는 폴더에서 다음 명령어를 실행합니다.

python 01_Linux-and-Python.py

터미널에서 Python 기초 과제와 학생 성적 분석 결과를 확인합니다.

<br />
<br />

#### 5. 참고 사항

모든 과제 및 프로젝트 코드를 하나의 Python 파일로 통합했습니다.

NumPy를 사용하므로 파일 상단에 import numpy as np가 필요합니다.

학생 이름과 성적 데이터는 코드 내부에 정의되어 있습니다.

학생별 평균 점수는 소수점 둘째 자리까지 출력했습니다.