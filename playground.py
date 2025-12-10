class Question:
    def __init__(self, a):
        self.a = a
    def quest(self):
        print(f"{self.a}")

# 클래스 인스턴스

ask = Question("안녕")  # self.a = "안녕"
ask.quest()

class sub_Question(Question):
    print(1)
print(2)
sub_a = sub_Question("hi")
sub_a.quest()

