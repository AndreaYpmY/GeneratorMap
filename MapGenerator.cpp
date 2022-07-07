/*
            PROGETTO "Algoritmi Paralleli e Sistemi Distribuiti" 2021/2022
            Andrea Tocci 

            Compile     : mpiCC MapGenerator.cpp -lallegro -lallegro_primitives
            Run         : mpirun -oversubscribe -np 4 ./a.out
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>
#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>

#define TITLE "Map Generator"
#define funzindex(r,c) ((r)*size+(c)) //funzione per calcolare posizione

int size=132;  
int *mr;
int *mw;
int *matrix;
int rank, num_Threads,rowSubMatrix,maxSteps=100;
MPI_Status status;
MPI_Request request;
MPI_Comm MAP;
int left,right;
MPI_Datatype MPI_SUB_MATRIX_SIZE;
MPI_Datatype MPI_ROW_SIZE;


ALLEGRO_DISPLAY * display;
ALLEGRO_EVENT_QUEUE *queue;
ALLEGRO_EVENT *event;
int molDisplay=5; //dimensioni, consigliato 5(molDisplay), 4(pixDisplay)
int pixDisplay=4;

inline void destroy();
inline void printAllegro();
inline void swap();
inline void changeCell(int x,int y,int stato);
inline void init();
inline void transInside();
inline void transBorder();
inline void sendBord();
inline void receiveBord();


int main(int argc, char **argv){

    MPI_Init(&argc, &argv);               
    MPI_Comm_size(MPI_COMM_WORLD, &num_Threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    while(size--%num_Threads!=0){} ++size; 
    rowSubMatrix=size/num_Threads;
    MPI_Type_contiguous(size,MPI_INT,&MPI_ROW_SIZE);
    MPI_Type_contiguous((size*(size/num_Threads)),MPI_INT,&MPI_SUB_MATRIX_SIZE);
    MPI_Type_commit(&MPI_SUB_MATRIX_SIZE);
    MPI_Type_commit(&MPI_ROW_SIZE);

    int period[1]={1};
    int dim[1]={num_Threads};
    MPI_Cart_create( MPI_COMM_WORLD , 1 , dim , period , 0 , &MAP);
    MPI_Cart_shift( MAP , 0 , 1 , &left , &right);

    matrix=new int [(size*size)];
    mr=new int [(rowSubMatrix+2)*size];
    mw=new int [(rowSubMatrix+2)*size];
    init();
    
    if(rank==0){
        
        al_init();
        display=al_create_display(size*molDisplay,size*molDisplay);
        al_init_primitives_addon();
        queue = al_create_event_queue();
        al_register_event_source(queue, al_get_display_event_source(display));
        if(!al_init()){
            printf("Failed to initialize allegro \n");
            destroy();
            MPI_Abort(MAP,777);
        }
        al_set_window_title(display, TITLE);
        
        matrix=new int[size*size];
    }

    for(int i=0;i<maxSteps;i++){
        sendBord();
        transInside();
        receiveBord();
        transBorder();
        swap();
        MPI_Gather(&mr[funzindex(1,0)], 1, MPI_SUB_MATRIX_SIZE, &matrix[funzindex(0,0)], 1, MPI_SUB_MATRIX_SIZE, 0, MAP);
        if(rank==0){
            printAllegro();
        }
    }
    
    destroy();
    return 0;
}

inline void destroy(){
    delete [] mr;
    delete [] mw;
    mr=mw=0;
    if(rank==0){
        delete[] matrix;
        matrix=0;
        al_destroy_display(display);
        al_destroy_event_queue(queue);
        al_uninstall_system();
    }
    MPI_Finalize();
}

inline void printAllegro(){
    al_clear_to_color(al_map_rgb(0, 0, 0));
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            if(matrix[funzindex(i,j)]==0) // Water 
                al_draw_filled_rectangle(i * molDisplay, j * molDisplay, i * molDisplay + pixDisplay, j * molDisplay + pixDisplay, al_map_rgb(0,0,255));  
            else if(matrix[funzindex(i,j)]==1) //Land
                al_draw_filled_rectangle(i * molDisplay, j * molDisplay, i * molDisplay + pixDisplay, j * molDisplay + pixDisplay, al_map_rgb(0,255,0)); 
            else if(matrix[funzindex(i,j)]==2) //Mountain
                al_draw_filled_rectangle(i * molDisplay, j * molDisplay, i * molDisplay + pixDisplay, j * molDisplay + pixDisplay, al_map_rgb(112,66,20));
            else if(matrix[funzindex(i,j)]==3) //Deep water
                al_draw_filled_rectangle(i * molDisplay, j * molDisplay, i * molDisplay + pixDisplay, j * molDisplay + pixDisplay, al_map_rgb(0,49,83));
            else if(matrix[funzindex(i,j)]==4) //Snow
                al_draw_filled_rectangle(i * molDisplay, j * molDisplay, i * molDisplay + pixDisplay, j * molDisplay + pixDisplay, al_map_rgb(255,255,255));
            else if(matrix[funzindex(i,j)]==5)//Forest
                al_draw_filled_rectangle(i * molDisplay, j * molDisplay, i * molDisplay + pixDisplay, j * molDisplay + pixDisplay, al_map_rgb(34,139,34));
        }
    }
    al_flip_display();
    al_rest(10.0 / 30.0);
}
    
inline void swap(){
    int *p=mr;
    mr=mw;
    mw=p;
}

inline void changeCell(int x,int y,int stato){
    int J;
    int contT=0,contM=0,contA=0,contAP=0,contN=0,contF=0; //contatori per ogni rispettivo stato
    /*
        contT   = Land
        contN   = Snow
        contM   = Mountain
        contA   = Water
        contAP  = Deep water
        contF   = Forest
    */
    for(int i=x-1;i<=x+1;i++){
        for(int j=y-1;j<=y+1;j++){ 
            if((i!=x || j!=y) && i>=0 && i<=rowSubMatrix+1&&j>=-1&& j<=size){
                J=j; 
                if(j==-1){ J=size-1;}
                int J=j%(size);
                if(mr[funzindex(i,J)]==1)        contT++;
                if(mr[funzindex(i,J)]==4)        contN++;
                if(mr[funzindex(i,J)]==2)        contM++;
                if(mr[funzindex(i,J)]==3)        contAP++;
                if(mr[funzindex(i,J)]==0)        contA++;
                if(mr[funzindex(i,J)]==5)        contF++;
            }
        }
    }
    unsigned *seedT=new unsigned(time(NULL)+rand()%100+x+y);
    int possibility=3+(rand_r(seedT)%100)%2;


    //se è Water
    if(stato==0 && contT==0 &&contM==0&&contF==0){
        mw[funzindex(x,y)]=3;
        return ;
    }else if(stato==0 && contT>3){
        mw[funzindex(x,y)]=1;
        return ;
    }else  if(stato==0 && (contF>3||contM>3||contN>0)){
        mw[funzindex(x,y)]=2;
        return ;
    }else if(stato==0){
        mw[funzindex(x,y)]=0;
        return ;
    }

    //se è Land
    if((stato==1) && contT+contF>=possibility){                        
        if(contT>=7 || contT+contF+contM==8 && contAP==0 &&contN==0 ){
            mw[funzindex(x,y)]=5; 
            return ;
        }
        mw[funzindex(x,y)]=1;
        return ;
    }
    else if(stato==1){
        mw[funzindex(x,y)]=0;
        return ;
    }

    //se è Mountain
    if(stato==2 && (contM==8||contM+contN==8)){
        possibility=rand_r(seedT)%100;
        if(possibility<=5)
            mw[funzindex(x,y)]=4;
        else
            mw[funzindex(x,y)]=2;
        return ;
    }else if(stato==2){
        mw[funzindex(x,y)]=2;
        return ;
    }

    //se è Deep water
    if(stato==3){
        if(contT>3)
            mw[funzindex(x,y)]=1;
        else if(contT==0 && contF==0 && contM==0) 
            mw[funzindex(x,y)]=3;
        else
            mw[funzindex(x,y)]=0;
        return ;
    } 

    //se è Snow
    if(stato==4){
        if(contN>2){
            mw[funzindex(x,y)]=0;
        }else
        {   
            mw[funzindex(x,y)]=4;  
        }
        return ;
    }

    //se è Forest
    if(stato==5){
        if(contF+contM+contN==8){
            mw[funzindex(x,y)]=2;
            return;
        }else if(contA==8){
            mw[funzindex(x,y)]=0;
        }else if(contA>4){ 
            mw[funzindex(x,y)]=1;
        }else{
            mw[funzindex(x,y)]=5;}
        return ;
    }  
}

inline void init(){
    unsigned *seedT=new unsigned(time(NULL)+rank);
    int terra=(int)(rowSubMatrix*size)/3;

    for(int i=0;i<rowSubMatrix+2;i++){
        for(int j=0;j<size;j++){
            mr[funzindex(i,j)]=0;
        }
    }
    for(int i=0;i<terra;i++){
        int x=1+rand_r(seedT)%((rowSubMatrix));
        int y=rand_r(seedT)%(size);
        while(mr[funzindex(x,y)]==1){
            x=1+rand_r(seedT)%((rowSubMatrix));
            y=rand_r(seedT)%(size);
        }
        mr[funzindex(x,y)]=1;
    }
}

inline void transInside(){
    for(int i=2;i<=rowSubMatrix;i++){
        for(int j=0;j<size;j++){
            changeCell(i,j,mr[funzindex(i,j)]);
        }
    }

}

inline void transBorder(){
    for(int i=1;i<=rowSubMatrix;i+=(rowSubMatrix-1)){
        for(int j=0;j<size;j++){
            changeCell(i,j,mr[funzindex(i,j)]);
        }
    }
}

inline void sendBord(){
    MPI_Isend(&mr[funzindex(1,0)],1,MPI_ROW_SIZE,left,777,MAP,&request);             
    MPI_Isend(&mr[funzindex(rowSubMatrix,0)],1,MPI_ROW_SIZE,right,666,MAP,&request); 
}

inline void receiveBord(){
    MPI_Recv(&mr[funzindex(rowSubMatrix+1,0)],1,MPI_ROW_SIZE,right,777,MAP,&status); 
    MPI_Recv(&mr[funzindex(0,0)],1,MPI_ROW_SIZE,left,666,MAP,&status);
}



