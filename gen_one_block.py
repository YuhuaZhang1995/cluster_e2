import numpy as np
import math
import scipy.optimize as opt
import sys
from collections import defaultdict

def gen_sample(alpha,theta,interactions,gen):
	if len(interactions)==0:
		interactions.append(1)
		s1=0
	else:
		p=(np.array(interactions)-alpha)/(sum(interactions)+theta) #sample an observed sender
		p=np.append(p,1-sum(p))
		#print(p)
		s1=np.random.multinomial(1,p)
		s1=np.where(s1==1)[0][0]
		
		if len(p)==(s1+1):
			interactions.append(1)
		else:
			interactions[s1]+=1

	p=(np.array(interactions)-alpha)/(sum(interactions)+theta) #sample an observed sender
	p=np.append(p,1-sum(p))
	s2=np.random.multinomial(1,p)
	s2=np.where(s2==1)[0][0]
		
	if len(p)==(s2+1):
		interactions.append(1)
	else:
		interactions[s2]+=1
	gen.append([s1,s2,0,0])

interactions=[]
gen=[]
for i in range(0,10000):
	gen_sample(0.5,10,interactions,gen)
print(interactions)
filename=sys.argv[1]
with open(filename,"a") as myfile:
	tmp="node1\tnode2\tc1\tc2\n"
	myfile.write(tmp)
	for i in gen:
		sender=i[0]
		receiver=i[1]
		tmp=str(sender)+"\t"+str(receiver)+"\t"+str(i[2])+"\t"+str(i[3])+"\n"
		myfile.write(tmp)
myfile.close()



