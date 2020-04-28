import numpy as np
import math
import scipy.optimize as opt
import sys

def llk(num_vertex,total_deg,k_deg,x):
	"""k_deg=a disctionary with key=degree, value=count of vertex"""
	"""num_vertex= number of vertex"""
	"""total_degree=total degree, 2*nrow"""
	"""x=[alpha,theta]"""
	alpha=x[0]
	theta=x[1]
	keys=np.array(list(dict.keys(k_deg)))
	values=np.array(list(dict.values(k_deg)))
	tmp=np.array(list(map(math.lgamma,(keys-alpha))))-math.lgamma(1-alpha)
	llk_=num_vertex*math.log(alpha)+math.lgamma(theta/alpha+num_vertex)-math.lgamma(theta/alpha)-math.lgamma(theta+total_deg)+math.lgamma(theta)+sum(np.multiply(values,tmp))
	return 0-llk_

def llk_der(num_vertex,total_deg,k_deg,x):
	"""The Jacobian required by L-BFGS-B algorithm"""
	alpha=x[0]
	theta=x[1]
	der=np.zeros_like(x)
	f1=lambda x: (-theta/alpha**2)/(theta/alpha+x)
	f2=lambda x: 1/(1-alpha+x)
	keys=np.array(list(dict.keys(k_deg)))
	values=np.array(list(dict.values(k_deg)))
	f3=lambda key: sum(f2(x) for x in range(key))
	tmp=np.array(list(map(f3,keys-1)))
	der[0]=0-(num_vertex/alpha+sum(f1(x) for x in range(num_vertex))-sum(np.multiply(values,tmp)))
	f4=lambda x: (1/alpha)/(theta/alpha+x)
	f5=lambda x: 1/(theta+x)
	der[1]=0-(sum(f4(x) for x in range(num_vertex))-sum(f5(x) for x in range(total_deg)))
	return der

def get_llkoptim(num_vertex,total_deg,k_deg,max_iter,x0):
	"""Get the minimal of llk"""
	"""Using L-BFGS-B algorithm"""
	f=lambda x: llk(num_vertex,total_deg,k_deg,x)
	f2=lambda x: llk_der(num_vertex,total_deg,k_deg,x)
	result=minimize(f,x0,method='L-BFGS-B',jac=f2,options={'maxiter':max_iter,'ftol':1e-5},bounds=((0.001,0.999),(0.001,100)))
	return result
	
#count the number of senders
remove_header=0
obs=[]
s1_list=[]
s2_list=[]
count_all={}
with open(filename) as infile:
	for line in infile:
		remove_header=remove_header+1
		s1=line.strip().split("\t")[0]
		s2=line.strip().split("\t")[1]
		#if change the sender, restart the loop
		if remove_header>1:
			obs.append([int(s1),int(s2)])
			if not s1 in s1_list:
				s1_list.append(s1)
				count_all[s1]=1
			else:
				count_all[s1]+=1
			if not s2 in s2_list:
				s2_list.append(s2)
				if not s2 in count_all:
					count_all[s2]=1
			else:
				count_all[s2]+=1
infile.close()

# Given value of alpha_zero and K
alpha_zero=[1,1]
K=2

# Initialize B, alpha_C, theta_C
B=np.matrix('0.5,0.5;0.5,0.5')
alpha_C=[0.1,0.1]
theta_C=[1,1]

# Initialize C_s
N1=len(s1_list) #number of unique sender 1
N2=len(s2_list) #number of unique sender 2
C_s1=np.random.multinomial(N1,[0.5,0.5])
C_s2=np.random.multinomial(N2,[0.5,0.5])
N=len(count_all)
C_s=np.random.multinomial(N,[0.5,0.5])
s_list=count_all.keys()
s_list=s_list.sort()

#Gibbs sampling
for iter in range(0,100):
	#update C_s
	for s in range(0,N):
		tmp_p=np.zeros(K)
		for k in range(0,K):
			tmp_p[k]=(alpha_zero[k]+len(np.where(C_s==k)[0])-1)/(sum(alpha_zero)+len(C_s)-1)
			#calculate P(S|C,alpha,theta)
			p=(len(np.where(np.array(s_list)==s)[0])-alpha_C[k])/(len(s_list)+theta_C[k]) 
			if p<0:
				p=(N*alpha_C[k]+theta_C[k])/(len(s_list)+theta_C[k])
			tmp_p[k]=tmp_p[k]*p
		tmp_p=np.array(tmp_p)/sum(tmp_p)
		C_s[s]=np.random.multinomial(1,tmp_p)
	
	#update alpha_c and theta_c
	for k in range(0,K):
		x0=[0.5,1]
		num_vertex=len(np.where(C_s==k)[0])
		s_tmp=s_list[np.where(C_s==k)[0]]
		count_tmp= {tt: count_all[tt] for tt in s_tmp if tt in count_all}
		count={}
		for i in count_tmp.keys():
			if count_tmp[i] in count.keys():
				count[count_tmp[i]]+=1
			else:
				count[count_tmp[i]]=1

		total_deg=sum(count_tmp.values())

		alpha_C=get_llkoptim(num_vertex,total_deg,count,100,x0).x[0]
		theta_C=get_llkoptim(num_vertex,total_deg,count,100,x0).x[1]
	
	#update B
	for i in range(0,K):
		for j in range(0,K):
			#B[i,j]=len(obs[where(s1 in i),where(s2 in j)])/len(obs[where s1 in j])
			count1=0
			count2=0
			for mm in obs:
				s1=np.where(s_list==mm[0])
				s2=np.where(s_list==mm[1])
				if C_s[s1]==i:
					count1+=1
					if C_s[s2]==j:
						count2+=1
			B[i,j]=count2/count1
				




