import numpy as np
import math
import scipy.optimize as opt
from scipy.optimize import minimize
import sys
	
######Input file processing#########
######count the number of senders, the degree, etc#############
remove_header=0
obs=[]
s1_list=[]
s2_list=[]
count_all={}
cluster_ass={}
filename=sys.argv[1]
with open(filename) as infile:
	for line in infile:
		remove_header=remove_header+1
		s1=line.strip().split("\t")[0]
		s2=line.strip().split("\t")[1]
		cs1=line.strip().split("\t")[2]
		cs2=line.strip().split("\t")[3]
		#if change the sender, restart the loop
		if remove_header>1:
			obs.append([int(s1),int(s2)])
			if not int(s1) in s1_list:
				s1_list.append(int(s1))
				count_all[int(s1)]=1
				cluster_ass[int(s1)]=int(cs1)
			else:
				count_all[int(s1)]+=1
			if not int(s2) in s2_list:
				s2_list.append(int(s2))
				if not int(s2) in count_all:
					count_all[int(s2)]=1
				if not int(s2) in cluster_ass:
					cluster_ass[int(s2)]=int(cs2)
			else:
				count_all[int(s2)]+=1
infile.close()

################Initialization###############
# Given value of alpha_zero and K
alpha_zero=[1,1] #hyperparameter
K=2 #number of cluster
a=[10,10] #Prior of theta
b=[10,10] #prior of theta
c=[1,1] #Prior of alpha 
d=[1,1] #Prior of alpha
Bb=np.matrix('90,10;10,90') #Prior of propensity matrix B

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

# Truth
"""C_s=[]
for key in sorted(cluster_ass.keys()):
	C_s.append(cluster_ass[key])
C_s=np.array(C_s)"""

###############Gibbs sampling#############3
for iter in range(0,100):
	print([alpha_C,theta_C])
	print(B)
	#print(np.linalg.norm(C_s-C_s_ass))
	
	#update C_s
	for s in range(0,N):
		tmp_p=np.zeros(K)
		for k in range(0,K):
			#Calculate P(C_s|C_{-s})
			tmp_p[k]=(alpha_zero[k]+len(np.where(C_s==k)[0])-1)/(sum(alpha_zero)+len(C_s)-1)
			
			#calculate P(S|C,alpha,theta)
			s_tmp=np.array(s_list)[np.where(C_s==k)[0]] #senders from cluster k
			count_K= {tt: count_all[tt] for tt in s_tmp if tt in count_all}
			if (s_list[s] in count_K) and (count_K[s_list[s]]!=1):
				p=((count_K[s_list[s]])-alpha_C[k]-1)/(sum(list(count_K.values()))+theta_C[k]-1)
			else:
				N_k=len(np.where(C_s==k)[0])
				if (s_list[s] in count_K) and (count_K[s_list[s]]==1):
					N_k=N_k-1
				p=(N_k*alpha_C[k]+theta_C[k])/(sum(list(count_K.values()))+theta_C[k]-1)
			
			#P(C_s|C_{-s},S) propto P(C_s|C_{-s})P(S|C,alpha,theta)
			tmp_p[k]=tmp_p[k]*p
		
		#Normalize
		tmp_p=np.array(tmp_p)/sum(tmp_p)
		#print(tmp_p)
		C_s[s]=np.where(np.random.multinomial(1,tmp_p)==1)[0][0]
	
	#update alpha_c and theta_c
	for k in range(0,K):
		if not len(np.where(C_s==k)[0])==0:
			num_vertex=len(np.where(C_s==k)[0]) #Count the number of unique senders in cluster k
			s_tmp=np.array(s_list)[np.where(C_s==k)[0]] #Get the sender list in cluster k
			count_tmp= {tt: count_all[tt] for tt in s_tmp if tt in count_all} #Get the corresponding degrees
			count={} #Total degrees
			for i in count_tmp.keys():
				if count_tmp[i] in count.keys():
					count[count_tmp[i]]+=1
				else:
					count[count_tmp[i]]=1

			total_deg=list(count_tmp.values())
			total_deg=sum(total_deg)
			x=np.random.beta(theta_C[k]+1,total_deg-1) #Latent X \sim Beta(theta+1,total degree-1)
			y=0 #Latent Y \sim Bernoulli(theta-1/theta+alpha*i)
			for i in range(0,len(s_tmp)-1):
				y=y+np.random.binomial(1,(theta_C[k]/(theta_C[k]+alpha_C[k]*(i+1))))
			z=0 #Latent z \sim bernoulli(j-1/j-alpha)
			for i in count_tmp:
				for j in range(0,count_tmp[i]-1):
					z=z+1-np.random.binomial(1,(j/(j+1-alpha_C[k])))
			#finally sampling of alpha and theta
			alpha_C[k]=np.random.beta(c[k]+len(s_tmp)-1-y,d[k]+z)
			theta_C[k]=np.random.gamma(y+a[k],1/(b[k]-np.log(x)))
		else:
			alpha_C[k]=0.5
			theta_C[k]=1
	
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
			tmp_alpha.append(count2)
		B[i,:]=np.random.dirichlet(tmp_alpha+np.array(Bb[i,:])[0],1)[0]
				




