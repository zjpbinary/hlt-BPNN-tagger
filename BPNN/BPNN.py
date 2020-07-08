import numpy as np

#模型基于Bengio师爷03年的《A Neural Probabilistic Language Model》
class Network:
    def __init__(self, filename, windows, embedsize, hiddensize):
        self.windows = windows
        self.hiddensize = hiddensize
        self.embedsize = embedsize
        self.word2index = dict()
        self.index2word = dict()
        self.tag2index = dict()
        self.index2tag = dict()
        self.sents, self.tags = self.prepare(filename)
        self.C = np.random.randn(len(self.word2index), embedsize)
        self.b = np.random.randn(len(self.tag2index))
        self.U = np.random.randn(len(self.tag2index), hiddensize)
        self.H = np.random.randn(hiddensize, windows*embedsize)
        self.d = np.random.randn(hiddensize)

    #读取毛文本与标注
    def load_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as f1:
            lines = [line.strip().split() for line in f1.readlines()]
        sentences = []
        taglists = []
        sent = []
        taglist = []
        tagset = set()
        wordset = set()
        for line in lines:
            if line == []:
                sentences.append(sent)
                taglists.append(taglist)
                sent = []
                taglist = []
            else:
                sent.append(line[1])
                wordset.add(line[1])
                if line[3]=='NP':
                    taglist.append('NN')
                    tagset.add('NN')
                else:
                    taglist.append(line[3])
                    tagset.add(line[3])

        return sentences, taglists, tagset, wordset

    #建立索引，词典
    def prepare(self, filename):
        sentences, taglists, tagset, wordset = self.load_data(filename)
        for i, tag in enumerate(tagset):
            self.tag2index[tag] = i
            self.index2tag[i] = tag
        for i, word in enumerate(wordset):
            self.word2index[word] = i
            self.index2word[i] = word
        wordnum = len(self.word2index)
        self.word2index['BOS'] = wordnum
        self.index2word[wordnum] = 'BOS'
        self.word2index['UNK'] = wordnum+1
        self.index2word[wordnum+1] = 'UNK'
        sents, tags = self.indexed(sentences, taglists, self.windows)

        return sents, tags

    #索引化，并根据窗口padding
    def indexed(self, sentences, taglists, windows):
        indexsents = []
        for sent in sentences:
            line = ['BOS']*(windows-1)
            line.extend(sent)
            indexsents.append([self.word2index[elem] if elem in self.word2index.keys() else self.word2index['UNK'] for elem in line])
        indextags = []
        for line in taglists:
            indextags.append([-1]*(windows-1)+[self.tag2index[tag] for tag in line])

        return indexsents, indextags

    #实现softmax函数
    def softmax(self, y):
        exp_y = np.exp(y)
        sum_exp_y = np.sum(exp_y)
        return exp_y/sum_exp_y

    #定义单个句子的训练过程，前向与反向
    def train(self, sent, tag, etc):
        for i in range(self.windows-1, len(sent)):
            #正向
            x = np.zeros(self.windows*self.embedsize)
            for k in range(self.windows):
                x[k*self.embedsize:(k+1)*self.embedsize] = self.C[sent[i-k]]
            a = np.tanh(np.matmul(self.H, x)+self.d)
            y = np.matmul(self.U, a)+self.b
            p = self.softmax(y)
            #反向
            gradient_y = -p
            gradient_y[tag[i]] += 1
            #更新输出层
            gradient_a = np.zeros(np.shape(a))
            for j in range(len(p)):
                self.b[j] += etc*gradient_y[j]
                gradient_a += gradient_y[j]*self.U[j]
                self.U[j] += etc*gradient_y[j]*a
            #更新隐层
            gradient_o = np.zeros(np.shape(a))
            for k in range(np.size(a)):
                gradient_o[k] = (1-a[k]**2)*gradient_a[k]
            self.d += etc*gradient_o
            gradient_x = np.matmul(self.H.T, gradient_o)
            self.H += etc*np.matmul(gradient_o.reshape(-1,1), x.reshape(1, -1))
            #更新嵌入层
            for k in range(self.windows):
                self.C[sent[i-k]] += etc*gradient_x[k*self.embedsize:(k+1)*self.embedsize]
    #预测
    def predict(self, sent):
        tag = []
        for i in range(self.windows-1, len(sent)):
            #正向
            x = np.zeros(self.windows*self.embedsize)
            for k in range(self.windows):
                x[k*self.embedsize:(k+1)*self.embedsize] = self.C[sent[i-k]]
            a = np.tanh(np.matmul(self.H, x)+self.d)
            y = np.matmul(self.U, a)+self.b
            p = self.softmax(y)
            tag.append(np.argmax(p))
        return tag

    #模型评估
    def evaluate(self, sents, tags):
        count = 0
        right = 0
        for i in range(len(sents)):
            pre = self.predict(sents[i])
            for j in range(len(pre)):
                count += 1
                if pre[j]==tags[i][j+self.windows-1]:
                    right += 1
        return right/count

    #开始训练
    def run(self, epoch, etc):
        for num in range(epoch):
            for i in range(len(self.sents)):

                self.train(self.sents[i], self.tags[i], etc)
            precision = self.evaluate(self.sents, self.tags)
            print('第%d迭代的精度为:'%num, precision)
            self.test('bigdata/test')
    #测试集测试
    def test(self, filename):
        sentences, taglists, _, _ = self.load_data(filename)
        sents, tags = self.indexed(sentences, taglists, self.windows)
        precision = self.evaluate(sents, tags)
        print('测试集上的精度为:', precision)


if __name__ == "__main__":
    model = Network('bigdata/train', 4, 256, 1024)
    print(len(model.sents))
    model.run(30, 0.00 1)
    model.test('bigdata/test')




