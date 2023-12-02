import paddle

class Att(paddle.nn.Layer):
    def __init__(self):
        super(Att,self).__init__()
        self.attention=paddle.nn.MultiHeadAttention(900,4)
        self.fc=paddle.nn.Linear(6,1)
        self.fla=paddle.nn.Flatten(1)

    def forward(self,x):
        x=self.attention(x)
        x=paddle.transpose(x,perm=[0,2,1])
        x=self.fc(x)
        x=self.fla(x)
        return x
    