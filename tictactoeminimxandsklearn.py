import cv2
from pylab import *
from scipy.ndimage import measurements, morphology
import numpy as np
import pickle
from sklearn.naive_bayes import BernoulliNB

with open('tictactoeminimxandsklearn.pkl', 'rb') as f:
    clf = pickle.load(f)


def problensolvernull(a,b,c,d,im):
    if( a<0 or b<0 or c<0 or d<0):
        return np.zeros((40,40))
    else:
        return im

def StartAnaliz(g):    
    try:
        len(g)>2 or len(g)<5
    except ValueError:
        print "Ошибка, таблица имеет длину", len(g)
            
    return len(g)
   


player='x'
computer='o'



def serchposition(grid, types=3):


    p=0
    c=0
    for i in range(len(grid)):
        p=0
        c=0
        for j in range(len(grid[i])):
            if(grid[i][j]=='x'):
                p+=1
            elif(grid[i][j]=='o'):
                c+=1
        if(p==types):
            return 10
        elif(c==types):
            return -10

        
    for i in range(len(grid)):
        p=0
        c=0
        for j in range(len(grid[i])):
            if(grid[j][i]=='x'):
                p+=1
            elif(grid[j][i]=='o'):
                c+=1
        if(p==types):
            return 10
        elif(c==types):
            return -10

    
    p=0
    c=0     
    for i in range(len(grid)):
        if(grid[i][i]=='x'):
            p+=1
        elif(grid[i][i]=='o'):
            c+=1
    if(p==types):
        return 10
    elif(c==types):
        return -10

    p=0
    c=0     
    for i in range(len(grid)):
        
        if(grid[(types-1)-i][i]=='x'):
              p+=1
        elif(grid[(types-1)-i][i]=='o'):
              c+=1
    if(p==types):
        return 10
    elif(c==types):
        return -10

    return 0   


def isMovesLeft(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if (board[i][j]=='_'):
                return True
    
    return False

def minimax(grid , gamers, deep):
    '''
    for i in  grid:  
        print i
      
    print '\n'
    '''
    if(serchposition(grid)==-10):
        return -10
    if(serchposition(grid)==10):
        return 10
   
    if(not isMovesLeft(grid)):
        return 0
    if(gamers==player):
        b=1000
        for i in range(len(grid)):
            for j in range(len(grid)):
                if(grid[i][j]=='_'):
                    grid[i][j]='o'
                    b=min(b,minimax(grid, computer, deep+1))
                    grid[i][j]='_'
        return b
    else:
        b=-1000
        for i in range(len(grid)):
            for j in range(len(grid)):
                if(grid[i][j]=='_'):
                    grid[i][j]='x'
                    b=max(b,minimax(grid, player, deep+1))
                    grid[i][j]='_'
        return b

def findBestMove(board,t):
    bestVal = -1000;
    stroka = -1;
    colonca = -1;



    for i in range(len(board)):
        for j in range(len(board[i])):
            if (board[i][j]=='_'):
                board[i][j]='x'
                moveVal = minimax(board, player,t)
                
                board[i][j] = '_'                
                if (moveVal > bestVal):
                    stroka = i;
                    colonca = j;
                    bestVal = moveVal;
                
 
  
    return  stroka, colonca


'''------------------------------------------------------------------------'''
    

def m_symbol(tip, x_grid,y_grid):
    t_1=0
    
    for i in range(len(x_grid)):
        if(i==int(tip[1])):
            for j in range(i,len(x_grid)):                
                if(x_grid[j]!=0):
                    t_1=x_grid[j]-1
                    break
                elif(j==len(x_grid)-1):
                    t_1=max(x_grid)
                    break
            break       
    t_2=0
    for i in range(len(y_grid)):
       if(i==int(tip[0])):
           for j in range(i,len(y_grid)):                
               if(y_grid[j]!=0):
                   t_2=y_grid[j]-1
                   break
               elif(j==len(y_grid)-1):
                   t_2=max(y_grid)
                   break
           break  



                
    return t_1  , t_2


def mesto_symbol(im):
    '''
     определяем местоположение символа
    '''
    symbol=measurements.center_of_mass(im)
    return symbol

def delete_max_pl(pl, im):
    '''
     удаление объекта с максимальной площадью
    '''
    m=max(pl)
    number=0
    for i in range(len(pl)):
        if(m==pl[i]):
            number=i+1
            break           
    labels, n= measurements.label(im)
    for x in range(len(im)):
        for y in range(len(im[x])):
            if(number==labels[x][y]):
                labels[x][y]=0;
        
    return labels

def ploshad_symbol(im):
    '''
    нахождение объекта с максимальной площалью
    '''
    labels, nbr= measurements.label(im)
    pl=np.zeros(nbr+1)
    for n in range(nbr+1):
        for x in range(len(labels)):
            for y in range(len(labels[x])):
                if(labels[x][y]==n):
                    pl[n]+=1
    
    return pl



def poisk_setki(im):
    '''
    ищем сетку
    '''
    labels, nbr= measurements.label(im)
    plochad=np.zeros(nbr+1)


    for z in range(1,nbr+1):    
        for x in range(len(labels)):
            for y in range(len(labels[x])):
                if(labels[x][y]==z):               
                    plochad[z]+=1 
    mximal_plochad=0
    for i in range(len(plochad)):
        if(plochad[i]== max(plochad)):
            mximal_plochad=i
        
    image=labels
    for x in range(len(labels)):
        for y in range(len(labels[x])):
            if(labels[x][y]!=mximal_plochad):
                image[x][y]=0
    
    return image


def poiskGradientaGrid(im, n):
    '''
    Вектора градиентов
    '''
    s=im.sum(axis=n)
    labels, nbr= measurements.label(s>(0.5*max(s)))
    return labels 

def elemGrid(im, n=0):
    '''
    Количество элементов
    '''
    s=im.sum(axis=n)
    labels, nbr= measurements.label(s>(0.5*max(s)))
    return nbr 

def Poisk_simbol(wisota,dlina, im):
    width=np.zeros(dlina)
    height=np.zeros(wisota)
    
    for i in range(wisota):
        for j in range(dlina):
            if(im[i][j]>0):
                width[j]+=1

    for i in range(dlina):
        for j in range(wisota):
            if(im[j][i]>0):
                height[j]+=1

    '''
    ширина
    '''
    n_1=0
    for i in range(len(width)):
        if(width[i]>0):
            n_1=i
            break
    n_2=0   
    for i in range(n_1,len(width)):
        if(width[i]==0):
           n_2=i
           break

    '''
    высота
    '''
    n_3=0
    for i in range(len(height)):
        if(height[i]>0):
           n_3=i
           break

    n_4=0   
    for i in range(n_3,len(height)):
        if(height[i]==0):
           n_4=i
           break              
  
    return problensolvernull(n_3-10,n_4+10,n_1-10,n_2+10,im[n_3-10:n_4+10,n_1-10:n_2+10])


im=cv2.imread("test.png",0)
img = cv2.medianBlur(im,5)
th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
im=(th1<128)*1
copy_im=im




'''
1.поиск сетки
1.1 создание векторов градиентов для поиска местоположения
2.создание матрицы сетки
3.уделение сетки
4.поиск символа
5.подача символа в класификтор
6.определение местоположения запись в матрицу
7.поиск места положения  на сетке
'''


obrazec=np.zeros(1601)


pl_grid=ploshad_symbol(copy_im)

x_grid=poiskGradientaGrid(poisk_setki(copy_im),0)
y_grid=poiskGradientaGrid(poisk_setki(copy_im),1)

matrix_grid=elemGrid(copy_im, n=0)

grid=np.chararray((matrix_grid+1,matrix_grid+1))
grid[:]="_"
matrix_image_delete_grid= delete_max_pl(pl_grid, im)


labels, nbr= measurements.label(matrix_image_delete_grid)
wisota=im.size/im[0].size
dlina=im[0].size

tipes=np.zeros((wisota,dlina))


for z in range(1,nbr+1):    
    for x in range(len(labels)):
        for y in range(len(labels[x])):
            if(labels[x][y]!=z):                
                tipes[x][y]=0
            else:
                tipes[x][y]=1
    mesto=np.array(measurements.center_of_mass(tipes))
   
    y1,x1 = m_symbol(mesto, x_grid ,y_grid)
    
    
    
    tests_im=Poisk_simbol(wisota,dlina, tipes)    
    image_resize=cv2.resize(tests_im,(40,40))      
    images_rocognition=image_resize.reshape(1600)
   
    
    for j in range(1600):
        obrazec[j]=0
        obrazec[j]=images_rocognition[j]

    number=clf.predict([obrazec])
    
    
    if(number==2):
        print 'void'
        print 'не раcпознан объект в ячейках',"горизонталь:", x1,"вертикаль:",y1, "\n"
    elif(number==1):
        print 'O'        
        grid[x1][y1]='o'
    else:
        print 'X'
        imshow(cv2.resize(tests_im,(40,40)))
        show()
        grid[x1][y1]='x'
        
   
    

    imshow(cv2.resize(tests_im,(40,40)))
    show()
    


'''
t=StartAnaliz(grid)
'''
print "\n"
print "Изначальная сетка"
print grid
print "\n"
print "начинаю расчет позиции...."
row,col = findBestMove(grid,StartAnaliz(grid))
grid[row][col]="x"
print "расчет позиции закончен"
for u in grid:
    print u    
print row,col



