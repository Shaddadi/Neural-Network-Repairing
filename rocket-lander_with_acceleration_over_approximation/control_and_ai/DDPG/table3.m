clc
clear

table1 = [ +0.063       3      332.7     +0.048            3      635.7     +0.053              2      446.1 ;                       
         +0.088       3      302.0     +0.012            6      1308.4    +0.085              3      1451.6    ;               
         +0.079       3      447.9     -0.084            4      812.9     -0.033              3      2417.1;
         +0.078       3      884.2     +0.025            3      620.3     +0.073              2      1395.3    ;            
         +0.085       3      754.3     -0.001            4      813.5     -0.165              5      2632.9    ];
     
     
table2 = [129.4     760.0       15.94       42.69      426.1       1583.6    13.38     36.39      700.5     2513.7   14.48    46.05;
         122.9     740.4       15.48       40.45      935.9       3352.8    13.58     37.23      618.6     2586.6   14.33    48.18;
	     206.7     1361.9      16.74       45.41      572.6       2106.7    13.55     36.55      645.3     3054.0   15.13    44.91 ;
	     329.0     2250.6      15.66       43.89      428.4       1569.7    13.67     35.97      714.4     3019.8   15.49    47.10;
	     224.53    1454.2      15.32       40.77      579.5       2108.3    13.68     35.99      997.0     3277.3    15.99   48.86 ]  ;
     
     
% computational
data1 = table2(:,[1,5,9,]);
data2 = table2(:,[2,6,10]);

data2./data1
mean(data2./data1, 'all')
     
% memory
data1 = table2(:,[3,7,11]);
data2 = table2(:,[4,8,12]);
1-data1./data2
mean(1-data1./data2, 'all')
     