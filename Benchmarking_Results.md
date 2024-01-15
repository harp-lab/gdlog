## Benchmarking result using MI250 (64GB) AMD GPU
```shell
# Benchmark SG
# fe_body
./SG ../data/data_163734.txt
Finished! sg_2__2_1 has 408443204
Join time: 10.2253 ; merge full time: 1.42593 ; rebuild full time: 0.573066 ; rebuild delta time: 0.0951821 ; set diff time: 5.56929
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.353924
sg counts 408443204
sg time: 19.574
join detail: 
compute size time:  0.378822
reduce + scan time: 0.0744815
fetch result time:  1.26583
sort time:          7.81807
build index time:   0
merge time:         0
unique time:        0.538546


# loc-Brightkite
./SG ../data/data_214078.txt
Finished! sg_2__2_1 has 92398050
Join time: 13.4307 ; merge full time: 0.0629133 ; rebuild full time: 0.0239498 ; rebuild delta time: 0.017839 ; set diff time: 0.432826
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.0280572
sg counts 92398050
sg time: 14.0023
join detail: 
compute size time:  0.255497
reduce + scan time: 0.0137241
fetch result time:  1.4384
sort time:          10.7955
build index time:   0
merge time:         0
unique time:        0.909317


# fe_sphere
./SG ../data/data_49152.txt
Finished! sg_2__2_1 has 205814096
Join time: 4.71153 ; merge full time: 0.673597 ; rebuild full time: 0.0638997 ; rebuild delta time: 0.0620844 ; set diff time: 2.87527
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.0791296
sg counts 205814096
sg time: 8.48438
join detail: 
compute size time:  0.235366
reduce + scan time: 0.0462961
fetch result time:  0.414899
sort time:          3.57669
build index time:   0
merge time:         0
unique time:        0.316202


# SF.cedge
./SG ../data/data_223001.txt
Finished! sg_2__2_1 has 382418182
Join time: 3.06879 ; merge full time: 2.38572 ; rebuild full time: 0.643336 ; rebuild delta time: 0.125202 ; set diff time: 8.59159
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.466628
sg counts 382418182
sg time: 20.5717
join detail: 
compute size time:  0.301976
reduce + scan time: 0.161965
fetch result time:  0.305877
sort time:          1.90091
build index time:   0
merge time:         0
unique time:        0.148628


# CA-HepTh
./SG ../data/data_51971.txt
Finished! sg_2__2_1 has 74618689
Join time: 5.67129 ; merge full time: 0.0304008 ; rebuild full time: 0.0188145 ; rebuild delta time: 0.0132835 ; set diff time: 0.165516
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.0217933
sg counts 74618689
sg time: 5.92582
join detail: 
compute size time:  0.113494
reduce + scan time: 0.00828013
fetch result time:  0.486698
sort time:          4.69365
build index time:   0
merge time:         0
unique time:        0.359771


# ego-Facebook
./SG ../data/data_88234.txt
Finished! sg_2__2_1 has 15018986
Join time: 2.76382 ; merge full time: 0.0058211 ; rebuild full time: 0.00392526 ; rebuild delta time: 0.00493294 ; set diff time: 0.0269965
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.00640189
sg counts 15018986
sg time: 2.81236
join detail: 
compute size time:  0.0629086
reduce + scan time: 0.00405999
fetch result time:  0.349504
sort time:          2.17792
build index time:   0
merge time:         0
unique time:        0.2201



# Benchmark Reachability
# com-dblp
./PLEN ../data/com-dblp.ungraph.txt
Iteration 4 finish populating
GPU 0 memory: free=41236299776, total=68702699520
Join time: 7.26062 ; merge full time: 0.173934 ; memory alloc time: 0.000529759 ; rebuild delta time: 0.0861753 ; set diff time: 0.286079
path_3__1_2_3   1 join result size(non dedup) 1310001323
GPUassert: out of memory /home/<USERNAME>/gdlog/gdlog/src/relation.hip 559
GPUassert: invalid argument /home/<USERNAME>/gdlog/gdlog/src/relation.hip 561
Memory access fault by GPU node-3 (Agent handle: 0x10ab1a0) on address 0x2d000. Reason: Write access to a read-only page.
[1]    191017 abort      ./PLEN ../data/com-dblp.ungraph.txt


# fe_ocean
./PLEN ../data/data_409593.txt
Iteration 246 finish populating
GPU 0 memory: free=1386217472, total=68702699520
Join time: 14.0213 ; merge full time: 24.1222 ; memory alloc time: 37.347 ; rebuild delta time: 0.324747 ; set diff time: 47.5378
Start merge full
GPUassert: out of memory /home/<USERNAME>/gdlog/gdlog/src/lie.hip 229
GPUassert: invalid argument /home/<USERNAME>/gdlog/gdlog/src/lie.hip 230
Memory access fault by GPU node-3 (Agent handle: 0x1ee3040) on address 0x18000. Reason: Write access to a read-only page.
[1]    206366 abort      ./PLEN ../data/data_409593.txt


# vsp_finan
./PLEN ../data/vsp_finan512_scagr7-2c_rlfddd.mtx
iteration 519 relation path_3__1_2_3 no new tuple added
Iteration 519 finish populating
GPU 0 memory: free=31524388864, total=68702699520
Join time: 6.90628 ; merge full time: 19.0522 ; memory alloc time: 1.77307 ; rebuild delta time: 0.266002 ; set diff time: 62.2861
Start merge full
GPUassert: out of memory /home/<USERNAME>/gdlog/gdlog/src/relation.hip 608
GPUassert: invalid argument /home/<USERNAME>/gdlog/gdlog/src/relation.hip 609
Memory access fault by GPU node-3 (Agent handle: 0x6f8040) on address 0x78000. Reason: Write access to a read-only page.
[1]    204812 abort      ./PLEN ../data/vsp_finan512_scagr7-2c_rlfddd.mtx



# p2p-Gnutella31
./PLEN ../data/data_147892.txt
Start merge full
GPUassert: out of memory /home/<USERNAME>/gdlog/gdlog/src/relation.hip 608
GPUassert: invalid argument /home/<USERNAME>/gdlog/gdlog/src/relation.hip 609
Memory access fault by GPU node-3 (Agent handle: 0x16c91a0) on address 0x32000. Reason: Write access to a read-only page.
[1]    84226 abort      ./PLEN ../data/data_147892.txt


# fe_body
./PLEN ../data/data_163734.txt
Finished! path_3__1_2_3 has 156120489
Join time: 1.73474 ; merge full time: 1.17083 ; rebuild full time: 0.0692892 ; rebuild delta time: 0.064017 ; set diff time: 3.00613
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.0751323
PLEN time: 6.14339
join detail: 
compute size time:  0.093474
reduce + scan time: 0.0221372
fetch result time:  0.226288
sort time:          1.18556
build index time:   0
merge time:         0
unique time:        0.13779

# SF.cedge
./PLEN ../data/data_223001.txt
Finished! path_3__1_2_3 has 80485066
Join time: 0.591755 ; merge full time: 0.761235 ; rebuild full time: 0.0400176 ; rebuild delta time: 0.0725839 ; set diff time: 2.02038
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.0700366
PLEN time: 3.57422
join detail: 
compute size time:  0.124276
reduce + scan time: 0.0362556
fetch result time:  0.092901
sort time:          0.213029
build index time:   0
merge time:         0
unique time:        0.0317306

# Benchmark CSPA
# httpd
./CSPA ../data/dataset/httpd
Finished! value_flow_2__1_2 has 1365306
Finished! value_flow_2__2_1 has 1365306
Finished! memory_alias_2__1_2 has 88905342
Finished! memory_alias_2__2_1 has 88905342
Finished! value_alias_2__1_2 has 234237608
Join time: 2.78842 ; merge full time: 0.270584 ; rebuild full time: 0.587309 ; rebuild delta time: 0.090212 ; set diff time: 0.711387
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.321214
analysis scc time: 6.74975
analysis scc time (chono): 6749
join detail: 
compute size time:  0.253036
reduce + scan time: 0.0462768
fetch result time:  0.527151
sort time:          2.10534
build index time:   0
merge time:         0.0494755
unique time:        0.31392


# linux
./CSPA ../data/dataset/linux
Finished! value_flow_2__1_2 has 5509641
Finished! value_flow_2__2_1 has 5509641
Finished! memory_alias_2__1_2 has 13777625
Finished! memory_alias_2__2_1 has 13777625
Finished! value_alias_2__1_2 has 30938106
Join time: 0.656022 ; merge full time: 0.047 ; rebuild full time: 0.0352189 ; rebuild delta time: 0.0390675 ; set diff time: 0.12887
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.0601689
analysis scc time: 1.3888
analysis scc time (chono): 1388
join detail: 
compute size time:  0.129463
reduce + scan time: 0.0249644
fetch result time:  0.140476
sort time:          0.224283
build index time:   0
merge time:         0.0235343
unique time:        0.0575929

# postgresql
./CSPA ../data/dataset/postgresql
Finished! value_flow_2__1_2 has 3712452
Finished! value_flow_2__2_1 has 3712452
Finished! memory_alias_2__1_2 has 89475479
Finished! memory_alias_2__2_1 has 89475479
Finished! value_alias_2__1_2 has 223793861
Join time: 2.18495 ; merge full time: 0.264613 ; rebuild full time: 0.644395 ; rebuild delta time: 0.0988435 ; set diff time: 0.638823
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.370552
analysis scc time: 6.79788
analysis scc time (chono): 6797
join detail: 
compute size time:  0.254576
reduce + scan time: 0.0490552
fetch result time:  0.301961
sort time:          1.49019
build index time:   0
merge time:         0.0360746
unique time:        0.244392
```
## Benchmarking result using MI50 (32GB) AMD GPU
```shell
# JLSE MI50 Bench
# Benchmark SG
# fe_body
./SG ../data/data_163734.txt
sg counts 408443204
sg time: 41.9938
join detail:
compute size time:  0.521996
reduce + scan time: 0.0909334
fetch result time:  3.12025
sort time:          14.2478
build index time:   0
merge time:         0
unique time:        1.53447

# loc-Brightkite
./SG ../data/data_214078.txt
sg counts 92398050
sg time: 30.0532
join detail:
compute size time:  0.446892
reduce + scan time: 0.0205751
fetch result time:  2.29054
sort time:          22.8577
build index time:   0
merge time:         0
unique time:        1.97498

# fe_sphere
./SG ../data/data_49152.txt
sg counts 205814096
sg time: 19.426
join detail:
compute size time:  0.255873
reduce + scan time: 0.0603159
fetch result time:  1.29861
sort time:          6.00981
build index time:   0
merge time:         0
unique time:        0.748529

# SF.cedge
./SG ../data/data_223001.txt
sg counts 382418182
sg time: 39.8404
join detail:
compute size time:  0.183431
reduce + scan time: 0.104463
fetch result time:  1.02499
sort time:          2.7993
build index time:   0
merge time:         0
unique time:        0.188131

# CA-HepTh
./SG ../data/data_51971.txt
sg counts 74618689
sg time: 13.4704
join detail:
compute size time:  0.203727
reduce + scan time: 0.0125887
fetch result time:  0.874423
sort time:          10.2675
build index time:   0
merge time:         0
unique time:        0.769815

# ego-Facebook
./SG ../data/data_88234.txt
sg counts 15018986
sg time: 6.31611
join detail:
compute size time:  0.0641238
reduce + scan time: 0.00486128
fetch result time:  0.591257
sort time:          4.43022
build index time:   0
merge time:         0
unique time:        0.487054

# Benchmark Reachability
# com-dblp
./PLEN ../data/com-dblp.ungraph.txt
Iteration 3 finish populating
GPU 0 memory: free=19610468352, total=34342961152
Join time: 3.54967 ; merge full time: 0.115742 ; memory alloc time: 0.00039824 ; rebuild delta time: 0.869942 ; set diff time: 0.164343
path_3__1_2_3   1 join result size(non dedup) 947674321
GPUassert: out of memory /home/ac.ashovon/gdlog/gdlog/src/join.hip 113
GPUassert: invalid argument /home/ac.ashovon/gdlog/gdlog/src/join.hip 114
Memory access fault by GPU node-3 (Agent handle: 0xf5d890) on address 0x36c1000. Reason: Page not present or supervisor privilege.
[1]    107812 abort      ./PLEN ../data/com-dblp.ungraph.txt


# fe_ocean
./PLEN ../data/data_409593.txt
iteration 59 relation path_3__1_2_3 finish dedup new tuples : 15077699 delta tuple size: 15077699 full counts 830398811
Iteration 59 finish populating
GPU 0 memory: free=6914310144, total=34342961152
Join time: 14.5125 ; merge full time: 4.1851 ; memory alloc time: 0.0103362 ; rebuild delta time: 3.05356 ; set diff time: 5.57887
path_3__1_2_3   1 join result size(non dedup) 41901279
Iteration 60 popluating new tuple
path_3__1_2_3
GPUassert: out of memory /home/ac.ashovon/gdlog/gdlog/src/relation.hip 473
Memory access fault by GPU node-3 (Agent handle: 0x1da4890) on address 0x1f000. Reason: Page not present or supervisor privilege.
[1]    109275 abort      ./PLEN ../data/data_409593.txt


# vsp_finan
./PLEN ../data/vsp_finan512_scagr7-2c_rlfddd.mtx
GPU 0 memory: free=6721372160, total=34342961152
Join time: 55.538 ; merge full time: 19.3639 ; memory alloc time: 0.0823746 ; rebuild delta time: 2.37231 ; set diff time: 43.4235
path_3__1_2_3   1 join result size(non dedup) 1512048
Iteration 277 popluating new tuple
path_3__1_2_3
GPUassert: out of memory /home/ac.ashovon/gdlog/gdlog/src/relation.hip 473
Memory access fault by GPU node-3 (Agent handle: 0x1deb890) on address 0x5e000. Reason: Page not present or supervisor privilege.
[1]    108316 abort      ./PLEN ../data/vsp_finan512_scagr7-2c_rlfddd.mtx

# p2p-Gnutella31
./PLEN ../data/data_147892.txt
Iteration 6 finish populating
GPU 0 memory: free=17685282816, total=34342961152
Join time: 3.42349 ; merge full time: 0.20545 ; memory alloc time: 0.0006392 ; rebuild delta time: 1.07349 ; set diff time: 0.31771
path_3__1_2_3   1 join result size(non dedup) 569112417
GPUassert: out of memory /home/ac.ashovon/gdlog/gdlog/src/relation.hip 559
GPUassert: invalid argument /home/ac.ashovon/gdlog/gdlog/src/relation.hip 561
Memory access fault by GPU node-3 (Agent handle: 0x1300890) on address 0x27000. Reason: Page not present or supervisor privilege.
[1]    107946 abort      ./PLEN ../data/data_147892.txt

# fe_body
./PLEN ../data/data_163734.txt
Finished! path_3__1_2_3 has 156120489
Join time: 8.44274 ; merge full time: 1.57938 ; rebuild full time: 0.108152 ; rebuild delta time: 0.832899 ; set diff time: 4.17726
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.866479
PLEN time: 15.276
join detail: 
compute size time:  0.0438669
reduce + scan time: 0.0267134
fetch result time:  0.513851
sort time:          1.47645
build index time:   0
merge time:         0
unique time:        0.128398

# SF.cedge
./PLEN ../data/data_223001.txt
Finished! path_3__1_2_3 has 80485066
Join time: 5.0825 ; merge full time: 1.22238 ; rebuild full time: 0.0608996 ; rebuild delta time: 0.224552 ; set diff time: 3.41672
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.23078
PLEN time: 10.133
join detail: 
compute size time:  0.0326515
reduce + scan time: 0.0322976
fetch result time:  0.185457
sort time:          0.307386
build index time:   0
merge time:         0
unique time:        0.0388818


# Benchmark CSPA
# httpd
./CSPA ../data/dataset/httpd
Finished! value_flow_2__1_2 has 1365306
Finished! value_flow_2__2_1 has 1365306
Finished! memory_alias_2__1_2 has 88905342
Finished! memory_alias_2__2_1 has 88905342
Finished! value_alias_2__1_2 has 234237608
Join time: 7.57668 ; merge full time: 0.425961 ; rebuild full time: 0.233271 ; rebuild delta time: 0.975249 ; set diff time: 1.1691
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 1.12203
analysis scc time: 15.2664
analysis scc time (chono): 15266
join detail: 
compute size time:  0.30577
reduce + scan time: 0.0622082
fetch result time:  1.62456
sort time:          3.71788
build index time:   0
merge time:         0.0723712
unique time:        0.566881

# linux
./CSPA ../data/dataset/linux
Finished! value_flow_2__1_2 has 5509641
Finished! value_flow_2__2_1 has 5509641
Finished! memory_alias_2__1_2 has 13777625
Finished! memory_alias_2__2_1 has 13777625
Finished! value_alias_2__1_2 has 30938106
Join time: 1.44817 ; merge full time: 0.0692388 ; rebuild full time: 0.0444058 ; rebuild delta time: 0.193057 ; set diff time: 0.239547
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.217464
analysis scc time: 3.31679
analysis scc time (chono): 3316
join detail: 
compute size time:  0.0776974
reduce + scan time: 0.0282187
fetch result time:  0.500127
sort time:          0.379403
build index time:   0
merge time:         0.0310624
unique time:        0.0780284


# postgresql
./CSPA ../data/dataset/postgresql
Finished! value_flow_2__1_2 has 3712452
Finished! value_flow_2__2_1 has 3712452
Finished! memory_alias_2__1_2 has 89475479
Finished! memory_alias_2__2_1 has 89475479
Finished! value_alias_2__1_2 has 223793861
Join time: 6.14489 ; merge full time: 0.423239 ; rebuild full time: 0.236538 ; rebuild delta time: 0.958649 ; set diff time: 1.03355
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 1.10803
analysis scc time: 14.551
analysis scc time (chono): 14551
join detail: 
compute size time:  0.283483
reduce + scan time: 0.0653045
fetch result time:  1.10403
sort time:          2.48954
build index time:   0
merge time:         0.0489407
unique time:        0.349514

```
