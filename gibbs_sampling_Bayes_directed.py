import numpy as np
import math
import scipy.optimize as opt
from scipy.optimize import minimize
import sys
	
#count the number of senders
remove_header=0
obs=[]
s1_list=[]
s2_list=[]
count_all={}
filename=sys.argv[1]
with open(filename) as infile:
	for line in infile:
		remove_header=remove_header+1
		s1=line.strip().split("\t")[0]
		s2=line.strip().split("\t")[1]
		#if change the sender, restart the loop
		if remove_header>1:
			obs.append([int(s1),int(s2)])
			if not int(s1) in s1_list:
				s1_list.append(int(s1))
				count_all[int(s1)]=1
			else:
				count_all[int(s1)]+=1
			if not int(s2) in s2_list:
				s2_list.append(int(s2))
				if not int(s2) in count_all:
					count_all[int(s2)]=1
			else:
				count_all[int(s2)]+=1
infile.close()

# Given value of alpha_zero and K
alpha_zero=[1,1]
K=2
a=5
b=20
c=1
d=1
Bb=[0.1,0.1]

# Initialize B, alpha_C, theta_C
B=np.matrix('0.5,0.5;0.5,0.5')
alpha_C=[0.1,0.1]
theta_C=[1,1]

# Initialize C_s
N=len(count_all)
C_s=[]
for i in range(0,N):
	C_s.append(np.where(np.random.multinomial(1,[0.5,0.5])==1)[0][0])
C_s=np.array(C_s)
s_list=list(count_all.keys())
s_list.sort()

#Gibbs sampling
for iter in range(0,200):
	print([alpha_C,theta_C])
	print(B)
	#print(C_s)
	#update C_s
	for s in range(0,N):
		tmp_p=np.zeros(K)
		for k in range(0,K):
			tmp_p[k]=(alpha_zero[k]+len(np.where(C_s==k)[0])-1)/(sum(alpha_zero)+len(C_s)-1)
			
			#calculate P(S|C,alpha,theta)
			s_tmp=np.array(s_list)[np.where(C_s==k)[0]]
			count_K= {tt: count_all[tt] for tt in s_tmp if tt in count_all}
			if (s_list[s] in count_K) and (count_K[s_list[s]]!=1):
				p=((count_K[s_list[s]])-alpha_C[k]-1)/(sum(list(count_K.values()))+theta_C[k]-1)
			else:
				N_k=len(np.where(C_s==k)[0])
				if (s_list[s] in count_K) and (count_K[s_list[s]]==1):
					N_k=N_k-1
				p=(N_k*alpha_C[k]+theta_C[k])/(sum(list(count_K.values()))+theta_C[k]-1)
			tmp_p[k]=tmp_p[k]*p
		
		tmp_p=np.array(tmp_p)/sum(tmp_p)
		#print(tmp_p)
		C_s[s]=np.where(np.random.multinomial(1,tmp_p)==1)[0][0]
	
	#update alpha_c and theta_c
	for k in range(0,K):
		#x0=[0.5,1]
		num_vertex=len(np.where(C_s==k)[0])
		s_tmp=np.array(s_list)[np.where(C_s==k)[0]]
		count_tmp= {tt: count_all[tt] for tt in s_tmp if tt in count_all}
		count={}
		for i in count_tmp.keys():
			if count_tmp[i] in count.keys():
				count[count_tmp[i]]+=1
			else:
				count[count_tmp[i]]=1

		total_deg=list(count_tmp.values())
		total_deg=sum(total_deg)
		x=np.random.beta(theta_C[k]+1,num_vertex-1)
		y=0
		for i in range(0,len(s_tmp)-1):
			y=y+np.random.binomial(1,(theta_C[k]/(theta_C[k]+alpha_C[k]*(i+1))))
		z=0
		for i in count_tmp:
			for j in range(0,count_tmp[i]-1):
				z=z+1-np.random.binomial(1,(j/(j+1-alpha_C[k])))
		alpha_C[k]=np.random.beta(c+len(s_tmp)-1-y,d+z)
		theta_C[k]=np.random.gamma(y+a,1/(b-np.log(x)))
	
	#update B
	for i in range(0,K):
		tmp_alpha=[]
		for j in range(0,K):
			#B[i,j]=len(obs[where(s1 in i),where(s2 in j)])/len(obs[where s1 in j])
			count1=0
			count2=0
			for mm in obs:
				s1=np.where(np.array(s_list)==mm[0])[0]
				s2=np.where(np.array(s_list)==mm[1])[0]
				if C_s[s1]==i:
					count1+=1
					if C_s[s2]==j:
						count2+=1
			tmp_alpha.append(count2/count1)
		B[i,:]=np.random.dirichlet(tmp_alpha+np.array(Bb),1)[0]
				




