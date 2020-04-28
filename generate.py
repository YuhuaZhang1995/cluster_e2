import numpy as np
import math
import scipy.optimize as opt
import sys
from collections import defaultdict

def gen_sample(alpha,theta,alpha_zero,B,interactions,generated_int):
	#Cluster assignment
	pi_s=np.random.dirichlet(alpha_zero)
	C_s1=np.random.multinomial(1,pi_s)
	C_s1=np.where(C_s1==1)[0][0]
	pi_s=np.random.dirichlet(alpha_zero)
	C_s2=np.random.multinomial(1,pi_s)
	C_s2=np.where(C_s2==1)[0][0]
	
	#generate the senders based on the DP
	if len(interactions[C_s1])==0:
		s1=0
		interactions[C_s1].append(1)

	else:
		p=(np.array(interactions[C_s1])-alpha[C_s1])/(sum(interactions[C_s1])+theta[C_s1]) #sample a new sample
		p=np.append(p,1-sum(p))
		s1=np.random.multinomial(1,p)
		s1=np.where(s1==1)[0][0]
		if len(interactions[C_s1])<=s1:
			interactions[C_s1].append(1)
		else:
			interactions[C_s1][s1]+=1
	
	if len(interactions[C_s2])==0:
		s2=0
		interactions[C_s2].append(1)

	else:
		p=(np.array(interactions[C_s2])-alpha[C_s2])/(sum(interactions[C_s2])+theta[C_s2]) #sample a new sample
		p=np.append(p,1-sum(p))
		s2=np.random.multinomial(1,p)
		s2=np.where(s2==1)[0][0]
		if len(interactions[C_s2])<=s2:
			interactions[C_s2].append(1)
		else:
			interactions[C_s2][s2]+=1

	
	#Given the cluster and the sender, decide whether to keep the sender in the file
	Flag=np.random.binomial(1,B[C_s1,C_s2])
	#print([C_s1,s1,C_s2,s2])
	if Flag==0 or (C_s1==C_s2 & s1==s2):
		if interactions[C_s2][s2]!=1:
			interactions[C_s2][s2]=interactions[C_s2][s2]-1
		else:
			interactions[C_s2].pop(s2)
		if interactions[C_s1][s1]!=1:
			interactions[C_s1][s1]=interactions[C_s1][s1]-1
		else:
			interactions[C_s1].pop(s1)
	else:
		generated_int.append([K*s1+C_s1,K*s2+C_s2])

alpha_zero=[1,1]
alpha=[0.2,0.8]
theta=[10,10]
B=np.matrix('0.9,0.1;0.1,0.9')
K=len(alpha_zero)
listKeys=list(range(0,K))
#interactions=dict(zip(listKeys, [[]]*len(listKeys))) #store the clusters and the nodes
interactions=defaultdict(list)
for i in listKeys:
    interactions[i]=[]

generated_int=[]
for i in range(0,1000):
	gen_sample(alpha,theta,alpha_zero,B,interactions,generated_int)

#print(interactions)
#print(generated_int)
with open("data_K_2.txt","a") as myfile:
	tmp="node1\tnode2\n"
	myfile.write(tmp)
	for i in generated_int:
		sender=i[0]
		receiver=i[1]
		tmp=str(sender)+"\t"+str(receiver)+"\n"
		myfile.write(tmp)
myfile.close()






