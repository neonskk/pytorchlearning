class person:
    def __call__(self, name):
        print("__call__"+" hello "+name)

    def hello(self,name):
        print("hello "+name)

xingming = person()
xingming("zhangsan")
xingming.hello("lisi")
# call函数更方便，不用.hello来调用方法