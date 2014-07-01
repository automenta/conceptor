patts{1} = @(n) sin(2 * pi * n / 10);
patts{2} = @(n) sin(2 * pi * n / 15);
patts{3} = @(n) sin(2 * pi * n / 20);
patts{4} = @(n) +(1 == mod(n,20));
patts{5} = @(n) +(1 == mod(n,10));
patts{6} = @(n) +(1 == mod(n,7));
patts{7} = @(n) 0;
patts{8} = @(n) 1;
rp = randn(1,4); 
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{9} = @(n) rp(mod(n,4)+1);
rp10 = rand(1,5);
maxVal = max(rp10); minVal = min(rp10); 
rp10 = 1.8 * (rp10 - minVal) / (maxVal - minVal) - 0.9;
patts{10} = @(n) (rp10(mod(n,5)+1));
rp = rand(1,6);
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{11} = @(n) (rp(mod(n,6)+1));
rp = rand(1,7);
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{12} = @(n) (rp(mod(n,7)+1));
rp = rand(1,8);
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{13} = @(n) (rp(mod(n,8)+1));
patts{14} = @(n) 0.5* sin(2 * pi * n / 10) + 0.5;
patts{15} = @(n) 0.2* sin(2 * pi * n / 10) + 0.7;
rp = randn(1,3);
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{16} = @(n) rp(mod(n,3)+1);
rp = randn(1,9);
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{17} = @(n) rp(mod(n,9)+1);
rp18 = randn(1,10);
maxVal = max(rp18); minVal = min(rp18); 
rp18 = 1.8 * (rp18 - minVal) / (maxVal - minVal) - 0.9;
patts{18} = @(n) rp18(mod(n,10)+1);
patts{19} = @(n) 0.8;
patts{20} = @(n) sin(2 * pi * n / sqrt(27));
patts{21} = @(n) sin(2 * pi * n / sqrt(19));
patts{22} = @(n) sin(2 * pi * n / sqrt(50));
patts{23} = @(n) sin(2 * pi * n / sqrt(75));
patts{24} = @(n) sin(2 * pi * n / sqrt(10));
patts{25} = @(n) sin(2 * pi * n / sqrt(110));
patts{26} = @(n) 0.1 * sin(2 * pi * n / sqrt(75));
patts{27} = @(n) 0.5 * (sin(2 * pi * n / sqrt(20)) + ...
    sin(2 * pi * n / sqrt(40)));
patts{28} = @(n) 0.33 * sin(2 * pi * n / sqrt(75));
patts{29} = @(n) sin(2 * pi * n / sqrt(243));
patts{30} = @(n) sin(2 * pi * n / sqrt(150));
patts{31} = @(n) sin(2 * pi * n / sqrt(200));
patts{32} = @(n) sin(2 * pi * n / 10.587352723);
patts{33} = @(n) sin(2 * pi * n / 10.387352723);
rp = rand(1,7);
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{34} = @(n) (rp(mod(n,7)+1));
patts{35} = @(n) sin(2 * pi * n / 12);
rpDiff = randn(1,5);
rp = rp10 + 0.2 * rpDiff;
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{36} = @(n) (rp(mod(n,5)+1));
patts{37} = @(n) sin(2 * pi * n / 11);
patts{38} = @(n) sin(2 * pi * n / 10.17352723);
patts{39} = @(n) sin(2 * pi * n / 5);
patts{40} = @(n) sin(2 * pi * n / 6);
patts{41} = @(n) sin(2 * pi * n / 7);
patts{42} = @(n) sin(2 * pi * n / 8);
patts{43} = @(n) sin(2 * pi * n / 9);
patts{44} = @(n) sin(2 * pi * n / 12);
patts{45} = @(n) sin(2 * pi * n / 13);
patts{46} = @(n) sin(2 * pi * n / 14);
patts{47} = @(n) sin(2 * pi * n / 10.8342522);
patts{48} = @(n) sin(2 * pi * n / 11.8342522);
patts{49} = @(n) sin(2 * pi * n / 12.8342522);
patts{50} = @(n) sin(2 * pi * n / 13.1900453);
patts{51} = @(n) sin(2 * pi * n / 7.1900453);
patts{52} = @(n) sin(2 * pi * n / 7.8342522);
patts{53} = @(n) sin(2 * pi * n / 8.8342522);
patts{54} = @(n) sin(2 * pi * n / 9.8342522);
patts{55} = @(n) sin(2 * pi * n / 5.1900453);
patts{56} = @(n) sin(2 * pi * n / 5.804531);
patts{57} = @(n) sin(2 * pi * n / 6.4900453);
patts{58} = @(n) sin(2 * pi * n / 6.900453);
patts{59} = @(n) sin(2 * pi * n / 13.900453);
rpDiff = randn(1,10);
rp = rp18 + 0.3 * rpDiff;
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{60} = @(n) (rp(mod(n,10)+1));
patts{61} = @(n) +(1 == mod(n,3));
patts{62} = @(n) +(1 == mod(n,4));
patts{63} = @(n) +(1 == mod(n,5));
patts{64} = @(n) +(1 == mod(n,6));
rp = randn(1,4); 
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{65} = @(n) rp(mod(n,4)+1);
rp10 = rand(1,5);
maxVal = max(rp10); minVal = min(rp10); 
rp10 = 1.8 * (rp10 - minVal) / (maxVal - minVal) - 0.9;
patts{66} = @(n) (rp10(mod(n,5)+1));
rp = rand(1,6);
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{67} = @(n) (rp(mod(n,6)+1));
rp = rand(1,7);
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{68} = @(n) (rp(mod(n,7)+1));
rp = rand(1,8);
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{69} = @(n) (rp(mod(n,8)+1));
rp = randn(1,4); 
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{70} = @(n) rp(mod(n,4)+1);
rp10 = rand(1,5);
maxVal = max(rp10); minVal = min(rp10); 
rp10 = 1.8 * (rp10 - minVal) / (maxVal - minVal) - 0.9;
patts{71} = @(n) (rp10(mod(n,5)+1));
rp = rand(1,6);
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{72} = @(n) (rp(mod(n,6)+1));
rp = rand(1,7);
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{73} = @(n) (rp(mod(n,7)+1));
rp = rand(1,8);
maxVal = max(rp); minVal = min(rp); 
rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
patts{74} = @(n) (rp(mod(n,8)+1));



% 1 = sine10  2 = sine15  3 = sine20  4 = spike20
% 5 = spike10 6 = spike7  7 = 0   8 = 1
% 9 = rand4; 10 = rand5  11 = rand6 12 = rand7
% 13 = rand8 14 = sine10range01 15 = sine10rangept5pt9
% 16 = rand3 17 = rand9 18 = rand10 19 = 0.8 20 = sineroot27
% 21 = sineroot19 22 = sineroot50 23 = sineroot75
% 24 = sineroot10 25 = sineroot110 26 = sineroot75tenth
% 27 = sineroots20plus40  28 = sineroot75third
% 29 = sineroot243  30 = sineroot150  31 = sineroot200
% 32 = sine10.587352723 33 = sine10.387352723
% 34 = rand7  35 = sine12  36 = 10+perturb  37 = sine11
% 38 = sine10.17352723  39 = sine5 40 = sine6
% 41 = sine7 42 = sine8  43 = sine9 44 = sine12
% 45 = sine13  46 = sine14  47 = sine10.8342522
% 48 = sine11.8342522  49 = sine12.8342522  50 = sine13.1900453
% 51 = sine7.1900453  52 = sine7.8342522  53 = sine8.8342522
% 54 = sine9.8342522 55 = sine5.19004  56 = sine5.8045
% 57 = sine6.49004 58 = sine6.9004 59 = sine13.9004
% 60 = 18+perturb  61 = spike3  62 = spike4 63 = spike5
% 64 = spike6 65 = rand4  66 = rand5  67 = rand6 68 = rand7
% 69 = rand8 70 = rand4  71 = rand5  72 = rand6 73 = rand7
% 74 = rand8