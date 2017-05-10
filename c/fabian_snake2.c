#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main( void )
{
  int num;
  int layer, maxnum_at_layer, offset_from_maxnum,median,offset_from_median;
  printf("Input the number:");
  scanf("%d",&num);

  layer = sqrt(num)+1;
  maxnum_at_layer = pow(layer,2);
  offset_from_maxnum = maxnum_at_layer - num;
  median = maxnum_at_layer - layer + 1;
  offset_from_median = abs(num-median);

  printf("layer %d, maxnum %d, offset from max %d, median %d, offset from median %d\n",layer,maxnum_at_layer,offset_from_maxnum,median,offset_from_median);

  if ( num == 1 )
    printf("%d %d",1,1);
  else if ( num == median )
    printf("%d %d",layer,layer);
  else if ( layer % 2 ==0 && num < median )
    printf("%d %d",layer-offset_from_median,layer);
  else if ( layer % 2 == 0 && num > median )
    printf("%d %d",layer,layer-offset_from_median);
  else if ( layer % 2 !=0 && num < median )
    printf("%d %d",layer,layer-offset_from_median);
  else if ( layer % 2 !=0 && num > median )
    printf("%d %d",layer-offset_from_median,layer);

  printf("\n");
   
}




