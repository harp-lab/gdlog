## Benchmarking result using MI250 (64GB) AMD GPU
```shell
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
Start merge full
GPUassert: out of memory /home/<USERNAME>/gdlog/gdlog/src/lie.hip 229
GPUassert: invalid argument /home/<USERNAME>/gdlog/gdlog/src/lie.hip 230
Memory access fault by GPU node-3 (Agent handle: 0xffe1a0) on address 0x3000. Reason: Unknown.
[1]    194401 abort      ./PLEN ../data/data_409593.txt

# vsp_finan
./PLEN ../data/vsp_finan512_scagr7-2c_rlfddd.mtx
num of sm 104
using 18446744073709551615 as empty hash entry
Input graph rows: 552020
[1]    195380 segmentation fault  ./PLEN ../data/vsp_finan512_scagr7-2c_rlfddd.mtx


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
# fe_body
./PLEN ../data/data_163734.txt
Finished! path_3__1_2_3 has 156120489
Join time: 8.43451 ; merge full time: 1.59695 ; rebuild full time: 0.108322 ; rebuild delta time: 0.834345 ; set diff time: 4.18308
Rebuild relation detail time : rebuild rel sort time: 0 ; rebuild rel unique time: 0 ; rebuild rel index time: 0.868321
PLEN time: 15.2892
join detail:
compute size time:  0.0448805
reduce + scan time: 0.0273218
fetch result time:  0.514161
sort time:          1.47523
build index time:   0
merge time:         0
unique time:        0.128438
```
