#include <stdio.h>


int main( int argc , char *argv[] )
{

  register int i;

  printf("%d\n",argc);
  
  for(i=0;i<argc;i++){
    printf("%s/",argv[i]);
  }
  printf("\n");

}
