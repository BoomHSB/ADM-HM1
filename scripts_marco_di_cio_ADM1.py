##Introduction/Say "Hello World!" with Python

print("Hello, World!")
      
##Introduction/If-Else

if n%2!=0:
    print("Weird")
elif n%2==0 and n>=2 and n<=5:
    print("Not Weird")
elif n%2==0 and n>=6 and n<=20:
    print("Weird")
else:
    print("Not Weird")
    
##Introdiction/Arithmetic Operators

n=int(input().strip())
a = int(input())
b = int(input())

print(a+b)
print(a-b)
print(a*b)
    

##Introduction/Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())

print(a//b)
print(a/b)

##Introduction/Loops

for i in range(0,n):
    print(i**2)
    
##Introduction/Write a function
        
def is_leap(year):
    if year%4==0 and year%100!=0:
        return True
    elif year%400==0:
        return True
    else:
        return False
        

##Introduction/ Print Function

#I consulted solution in this one as i didn't know how to print them one next to another.

print(*range(1,n+1), sep='')



##Basic Data Types/ Tuples

n=int(input())
t = tuple(int(x) for x in input().split())
print(hash(t))


##Basic Data Types / Find the runner-up score

n=int(input())
s=input()
g=s.split(" ")
for k in range(len(g)):
    g[k]=int(g[k])
m=max(g)
while m in g:
    g.remove(m)
print(max(g))

##Basic Data Types / Lists

test =int(input())
s=[]
for _ in range (test):
    cmd=list(input().split())
    
    if cmd[0]=='insert':
        s.insert(int(cmd[1]),int(cmd[2]))
    elif cmd[0]=="remove":
        s.remove(int(cmd[1]))
    elif cmd[0]=="append":
        s.append(int(cmd[1]))
    elif cmd[0]=="sort":
        s.sort()
    elif cmd[0]=="pop":
        s.pop()
    elif cmd[0]=="reverse":
        s.reverse()
    elif cmd[0]=="count":
        v=s.count(int(cmd[1]))
        print(v)
    elif cmd[0]=="index":
        x=s.index(int(cmd[1]))
        print(x)
   
    elif cmd[0]== 'print':
        print(s)

##Basic Data Types / Tuples

input()
hti = map(int, input().strip().split(" "))
print(hash(tuple(hti)))



         
##Strings / Designer Door Mat
  
 n, m = map(int,input().split())
pattern = [('.|.'*(2*i + 1)).center(m, '-') for i in range(n//2)]
print('\n'.join(pattern + ['WELCOME'.center(m, '-')] + pattern[::-1]))

## Strings / sWAP cASE
n=input()
print(n.swapcase())


##Strings / String Split and Join

a=input()
a=a.split(" ")
a="-".join(a)
print(a)
 
 
##Strings / What's your name?

a=input()
b=input()
print("Hello",a,b,"! You just delved into python")

##Strings / Mutations

#in this solution the user is asked to type something first
then he is asked to choose which character of whatever 
he typed will be replaced by the character k.

string = input()
l = list(string)
print(l) #for clarity purposes
i=int(input())
l[i] = "k"
string = ''.join(l)
print (string)


##Strings / String Validators

S=str(input())
print (S,"is alphanumeric: ",S.isalnum())
print (S,"is alphabetical: ",S.isalpha())
print (S,"is all digits: ",S.isdigit())
print (S,"is lowercase: ",S.islower())
print (S,"is uppercase: ",S.isupper())

##Strings / Text Alignment

##this required some trial and error(guessing) to make the bottom cone.


thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))




## Strings / Alphabet Rangoli ##HEELP

thickness = int(input())
c="a"
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
    

##Strings / The Minion Game

s = raw_input()

vowels = 'AEIOU'

k_score = 0
s_score = 0
for i in range(len(s)):
    if s[i] in vowels:
        kevsc += (len(s)-i)
    else:
        stusc += (len(s)-i)

if kevsc > stusc:
    print "Kevin", kevsc
elif kevsc < stusc:
    print "Stuart", stusc
else:
    print "Draw"

## Strings / Merge the Tools!

import random

s=input("type your string")
n=len(s)

factors=[]
for i in range(1,n+1):
    if n%i==0:
       factors.append(i)
       
print(factors)
d=random.choice(factors)

def dividestring(s,d):
    str_size=lens(s)
    part_size = str_size/d
    k = 0
    for i in string: 
        if k%part_size==0: 
            print ("\n"), 
        print (i), 
        k += 1
        
        
 
##Strings / String Formatting

def converti(n,b):
    if b!=16:
        if n<b:
            return str(n)
        else:
            lista_resti=[]
            k=n
            k=k//b
            r=k%b
            lista_resti.append(r)
            while k!=0:
                k=k//b
                r=k%b    
                lista_resti.append(r)
            print(lista_resti)
            lista_resti.reverse()
            return ''.join(lista_resti)
    else:
        if n<b:
            if n<10:
                return str(n)
            else:
                if n==10:
                    return 'A'
                elif n==11:
                    return 'B'
                elif n==12:
                    return 'C'
                elif n==13:
                    return 'D'
                elif n==14:
                    return 'E'
                elif n==15:
                    return 'F'
        else:
            lista_resti=[]
            k=n
            k=k//b
            r=k%b
            lista_resti.append(r)
            while k!=0:
                k=k//b
                r=k%b    
                lista_resti.append(r)
            lista_resti.reverse()
            for i in range(len(lista_resti)):
                if lista_resti[i]==10:
                    lista_resti[i]='A'
                elif lista_resti[i]==11:
                    lista_resti[i]='B'
                elif lista_resti[i]==12:
                    lista_resti[i]='C'
                elif lista_resti[i]==13:
                    lista_resti[i]='D'
                elif lista_resti[i]==14:
                    lista_resti[i]='E'
                elif lista_resti[i]==15:
                    lista_resti[i]='F'
            return ''.join(lista_resti)
            
 
def print_formatted(number):
    conv_bin=converti(number,2)
    np=len(conv_bin)
    for i in range(1,number):
        s=''
        s=s+str(i)+' '*(np-len(str(i)))
        conv_8=converti(i,8)
        conv_16=converti(i,16)
        conv_2=converti(i,2)
        s=s+conv_8+' '*(np-len(str(conv_8)))
        s=s+conv_16+' '*(np-len(str(conv_16)))
        s=s+conv_2
        print(s)
    
print_formatted(5)



##Sets/Introduction to Sets

k,array=int(input()),set(map(int,input().split()))
print(sum(myset)/len(array))

 
##Sets/Set .add()

num = int(input())
data = set()

for x in range(num):
  data.add(input())
  
print(len(data))


##Sets / Set .interesection()

firstline=[9]
secondline={1,2,3,4,5,6,7,8,9}
thirline=[9]
fourthline={10,1,2,3,11,21,55,6,8}

print(len(secondline.intersection(fourthline)))

#or with given input(to be inserted without commas or spacing)

a=set(input())
b=set(input())
 
bothsubs=len(a.intersection(b))



## Sets/ Symmetric Difference


a=[4]
b=[2,4,5,9]
c=[4]
d=[2,4,11,12]

a=set(a)
b=set(b)
c=set(c)
d=set(d)


x=a.symmetric_difference(b)
y=x.symmetric_difference(c)
solution= y.symmetric_difference(d)

for i in solution:
    print (i)
    




##Sets/Discard remove and pop
    
n=int(input())
s=[input()]
c=int(input)


n={9}
s=set([1,2,3,4,5,6,7,8,9])
c={10}

s.pop()
s.remove(9)
s.discard(9)
s.discard(8)
s.remove(7)
s.pop()
s.discard(6)
s.remove(5)
s.pop()
s.discard(5)

print(sum(s))

## Sets/Symmetric Difference operation
firstline=[9]
secondline={1,2,3,4,5,6,7,8,9}
thirline=[9]
fourthline={10,1,2,3,11,21,55,6,8}

print(len(secondline.symmetric_difference(fourthline)))



##Sets/Set Mutations

s1=set([1,2,3,4,5,6,7,8,9,10,11,12,13,14,24,52])
s2=set([2,3,5,6,8,9,1,4,7,11])
s3=set([55,66])
s4=set([22,7,35,62,58])
s5=set([11,22,35,55,58,62,66])

s1.intersection_update(s2)
s1.update(s3)
s1.symmetric_difference_update(s4)
s1.difference_update(s5)

print(sum(s1))


##Sets/The Captain's Room 

array1=[1,2,3,6,5,4,4,2,5,3,6,1,6,5,3,1,4,1,2,5,1,4,3,6,8,4,3,1,5,6,2]
length1=len(ar)
 
#we know that the only room number to appear just once in the array will be the captains room number, so we could 
define a function to find the least frequent element in an array.

def leastfreq(ar,n):
    ar.sort()
    min_count=n+1
    res=-1
    curr_count=1
    for i in range(1,n):
        if (ar[i]==ar[i-1]):
            curr_count=curr_count+1
        else:
            if (curr_count<min_count):
                min_count=curr_count
                res=ar[i-1]
            
            curr_count=1
        if (curr_count<min_count):
            min_count=curr_count
            res=ar[n-1]
            return res
            

     
     
leastfreq(array1,length1)

##Closures and Decorations /Standardize Mobile Numbers Using Decorators

pn=int(input())
lst=[]
for i in range(0,pn):
    a=input()
    lst.append(a[-10:])
lst=sorted(lst)

for item in lst:
    print("91",item[0:5],item[5:])
    

 
 
##Sets/Check Subset

for i in range(int(input())):
    a=int(input()); A=set(input().split())
    b=int(input()); B=set(input().split())
    print(A.issubset(B))
    
##Sets/Check strict Superset

a=set(input().split())
print(all(a> set(input().split()for i in range(int(input()))))


##Sets/Union Operation

a,A=int(input()),set(map(int,input().split()))
b,B=int(input()),set(map(int,input().split()))

print(len(A.union(B)))


##Sets/ Set Difference Operation

a,A=int(input()),set(map(int,input().split()))
b,B=int(input()),set(map(int,input().split()))

print(len(A.difference(B)))

##Sets/ No Idea

l1=set(input().split())
l2=set(input().split())
A=set(input().split())
B=set(input().split())
print(len(A.intersection(l2))-len(B.intersection(l2)))


## Collections/ DefaultDict Tutorial

from collections import defaultdict
n, m = input().split()
d = defaultdict(list)

for i in range(int(n)):
    d[input()].append(str(i+1))
    
for _ in range(int(m)):
    i = input()
    print(' '.join(d[i])) if d[i] else print(-1)



##Collections / Counter
    
from collections import Counter

X=int(input())
ss=list(map(int,input().split()))
N=int(input())
css= Counter(ss)
revenue=0
for i in range(N):
    sd,sp=map(int,input().split())
    if(css[sd]):
        css[sd]-= 1
        revenue +=sp
print(revenue)

##Collections / namedtuple

from collections import namedtuple

N, mark, avg = int(input()), input().split().index('MARKS'), 0
for i in range(N):
    avg += float(input().split()[mark])
print("{0:.2f}".format(avg/N)


##Collections / OrderedDict

from collections import OrderedDict

d = OrderedDict()
for _ in range(int(input())):
    item, space, quantity = input().rpartition(' ')
    d[item] = d.get(item, 0) + int(quantity)
for item, quantity in d.items():
    print(item, quantity)

##Collections / Deque

from collections import deque
d = deque()

for _ in range(int(input())):
    r = list(input().split())
    if r[0] == 'append':
        d.append(r[1])
    elif r[0] == 'appendleft':
        d.appendleft(r[1])
    elif r[0] == 'pop':
        d.pop()
    else:
        d.popleft()

print(" ".join(d))

##Collections / Word Order

count = int(input())
order = []
words = {}

for i in range(0, count):
    word = input()
    
    if word in words:
        words[word] += 1
    else:
        order.append(word)
        words[word] = 1

print(len(words))

for word in order:
    print(words[word], end=" ")
    
    
##Collections / Company Logo

import collections

c=str(input())
print(collections.Counter(c).most_common(3))
print (collections.Counter(c).keys())



##Date and Time / Calendar Module

import calendar
m, d, y = input().split()
print(calendar.day_name[calendar.weekday(int(y), int(m), int(d))].upper())



##Date and Time / Time Delta

#T had to consult solution as i couldn't find how to input the 
#time of the day in correct format

import datetime

format_string = "%a %d %b %Y %H:%M:%S %z"
T = int(input())

for _ in range(T):
    d1 = str(input())
    d2 = str(input())

    parsed_t1 = datetime.datetime.strptime(t1, format_string)
    parsed_t2 = datetime.datetime.strptime(t2, format_string)

    diff = parsed_t2 - parsed_t1

    print (int(abs(diff.total_seconds())))
    
    
##Exceptions

for i in range(int(input())):
    try:
        a,b=map(int,input().split())
        print(a//b)
    except Exception as e:
        print("Error Code:",e)

##Builtins / Zipped

n, x = map(int, input().split()) 

sheet = []
for _ in range(x):
    sheet.append( map(float, input().split()) ) 

for i in zip(*sheet): 
    print( sum(i)/len(i) )
    
    
##Builtins / Input

a,e=input()
l1=list(a)
x=float(l1[0])
y=float(l1[1])

e=x**3+x**2+x+1
print(e==y)

x,e=map(int, input().split())
y=input()
print(e==y)

#Builtins / Evaluation

eval(input())

## Builtins / Any or All

n = int(input())
lst = list(map(int, input().split()))
print(all(i>0 for i in lst) and any(str(i) == str(i)[::-1] for i in lst))


## Builtins / ginorts
s="sorting1234"


##Functionals / Map and Lambda function

def Fibonacci(n): 
    if n<0: 
        print("Incorrect input") 
   
    elif n==1: 
        return 0
    
    elif n==2: 
        return 1
    else: 
        return Fibonacci(n-1)+Fibonacci(n-2)
    


#Numpy / Numpy Arrays
import numpy
data = input().split(" ")
print(numpy.array(data[::-1], float))

#Numpy / shape and reshape 

a=numpy.array(list(map(int,input().split())))
print(numpy.reshape(a,(3,3)))


#Numpy / Transpoe and flatten
n,m=map(int,input().split())

l1=[list(map(int,input().split())) for i in range(n)]
a=numpy.array(lista)

print(numpy.transpose(a))
print(a.flatten())

#Numpy / Concatenate

n, m, p = map(int, input().split())
ar1 = [list(map(int, input().split())) for _ in range(n)]
ar2 = [list(map(int, input().split())) for _ in range(m)]
print(numpy.concatenate((ar1, ar2), axis=0))



#Numpy / Zeros and Ones

x = [int(i) for i in input().split(" ")]
print((numpy.zeros((x),dtype=numpy.int)))
print((numpy.ones((x),dtype=numpy.int)))

#Numpy / Eye and Identity
n, m = [int(i) for i in input().strip().split()]
print(numpy.eye(n,m,k=0))


#Numpy / Array Mathematics
import numpy

N, M = map(int, input().split())

A = numpy.array([list(map(int, input().split())) for n in range(N)])
B = numpy.array([list(map(int, input().split())) for n in range(N)])

print (A + 😎
print (A - 😎
print (A * 😎
print (A // 😎
print (A % 😎
print (A ** 😎
 
#NUmpy / Floor Ceil and Rint

arr=numpy.array(list(input().split()),float)
print(numpy.floor(arr))
print(numpy.ceil(arr))
print(numpy.rint(arr))

#Numpy / Sum and Prod

n,m=[int(a) for a in input().split()]

arr=numpy.array([input().split() for a in range(m)],int)
arr_sum=numpy.sum(arr,axis=0)
print(numpy.prod(arr_sum))

#Numpy / Min and Max

from numpy import array, min, max

N, M = map(int, input().split())
num = array([list(map(int, input().split())) for _ in range(N)])
print(max(min(num, axis = 1)))

#Numpy / Mean, Var, and Std

n,m = map(int,input().split())
arr = numpy.array([input().split() for _ in range(n)], int)

print(arr.mean(axis=1))
print(arr.var(axis=0))
print(arr.std())


#Numpy / Dot and Cross

n=int(input())
a=numpy.array([input().split() for i in range(n)],int)
b=numpy.array([input().split() for i in range(n)],int)
print(numpy.dot(a,b))

#Numpy / Inner and Outer
import numpy

a = numpy.array(list(map(int, input().split())))
b = numpy.array(list(map(int, input().split())))

print (numpy.inner(a,b))
print (numpy.outer(a,b))


##Numpy / Polinomials

import numpy
p = numpy.array(list(map(float,input().split())))
x = float(input())
print(numpy.polyval(p,x))


#Numpy / Linear Algebra
n = int(input())
a = numpy.array([input().split() for i in range(n)], float)
print(numpy.linalg.det(a)


 
###Homework 1 Second Set of Problems

#Birthday Candles

total_candles=int(input())
candle_heights=list(map(int,input().split()))
candles_blown=candle_heights.count(max(candle_heights))
print(candles_blown)

#Kangaroos

x1=int(input())
v1=int(input())
x2=int(input())
v2=int(input())
njumps=10^5


lst=[]
def kangaroo1(x1,v1,x2,v2):
    for i in range(1,njumps):
        if (x1+v1*i==x2+v2*i):
            lst=["YES"]
        else:
            lst=["NO"]
        
    return lst
kangarooun(0,3,4,2)
kangarooun(3,4,7,4)
kangarooun(0,1,2,4)


def kangaroo2(x1,v1,x2,v2):
   if any([(x1+v1*i==x2+v2*i) for i in njumps]):
       print("YES")
   else:
       print("NO")
       
def kangaroo3(x1,x2,x3,x4):
   if x1+v1*i==x2+v2*i for any([i i\n njumps]):
       print("YES")
   else:
       print("NO")
       


#Viral Advertising
       
n=int(input("number of days"))    
shared=[]
Liked=[]
ttlliked=[]  
def ViralAds(n):
    if n==1:
        shared=[0]
 