import jieba
import sys

reload(sys)

sys.setdefaultencoding('utf-8')

in_file_name = "atec_nlp_sim_train.csv"
out_file_name = "train.txt"

def process(inpath, outpath):
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            lineno, sen1, sen2, label = line.strip().split('\t')
            fout.write(lineno + '\t')
            for w in jieba.cut(sen1):
                if w.strip() and w!=('*'):
                    fout.write(w + ' ')
            fout.write('\t')
            for w in jieba.cut(sen2):
                if w.strip() and w!=('*'):
                    fout.write(w + ' ')
            fout.write('\t'+label+'\n')
            

process(in_file_name,out_file_name)