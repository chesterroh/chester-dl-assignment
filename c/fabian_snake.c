#include <stdio.h>

#define MAXLAYER 100000

void find_layer( int num, int *layer, int *maxnum_at_layer )
{
  register int i;
  int total=0;

  for(i=1 ; i < MAXLAYER; i++){
    total += (i*2-1);
    printf("total %d\n",total);

    if( total >= num ){
      *layer = i;
      *maxnum_at_layer = total;
      return;
    }
  }
}

int main( int argc , char **argv )
{
  int num=0;
  int layer,maxnum_at_layer;
  printf("Enter the number:");
  scanf("%d",&num);

  find_layer(num, &layer, &maxnum_at_layer);

  printf("%d-th layer's maxnum is %d, median value is %d\n",layer,maxnum_at_layer,maxnum_at_layer-layer+1);

}
