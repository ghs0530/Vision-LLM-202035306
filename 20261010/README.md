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

#### $5)$ NumPy

-1

NumPy 배열에 대해 정규화를 한 다는 것이 보통 Min-Max 정규화를 적용한다는 것을 검색해서 찾아 본 뒤에 코드로 작성했습니다.

import numpy as np

np_arr = np.array([1, 2, 3, 4, 5])

np_arr_normalized = (np_arr - np_arr.min()) / (np_arr.max() - np_arr.min())

print(np_arr_normalized)

실행 결과



-2

[0.   0.25 0.5  0.75 1.  ]

2.4 NumPy 배열을 float32로 변환

정규화한 배열을 astype() 함수를 사용해 float32 자료형으로 변환했습니다.

np_arr_final = ((np_arr - np_arr.min()) / (np_arr.max() - np_arr.min())).astype(np.float32)

print(np_arr_final)
print(np_arr_final.dtype)




#### 실행 환경 및 의존성

사용 언어: Python 3

개발 환경: Visual Studio Code (VS Code)

제출 파일 형식: .py

사용 라이브러리: NumPy

필요한 라이브러리 설치

Python 3이 설치된 환경에서 터미널을 열고 다음 명령어를 실행합니다.

pip install numpy

#### 실행 방법

Python 3을 설치합니다.

Visual Studio Code에서 과제 폴더를 엽니다.

터미널에서 pip install numpy 명령어를 실행하여 NumPy 라이브러리를 설치합니다.

제출한 01_Linux-and-Python.py 파일을 엽니다.

VS Code의 터미널에서 다음 명령어를 실행합니다.

python 01_Linux-and-Python.py

터미널에 출력되는 결과를 확인합니다.