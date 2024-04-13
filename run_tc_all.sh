echo "Preparing code and building TC"
git stash && git checkout main 
cmake -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -Bbuild . && cd build && make -j 
cd ..
echo ">>>>>>>>>>>>>>>>>> Testing REACH >>>>>>>>>>>>>>>>>"
echo " >>>>> Testing GDlog: "
echo " >>>>> Generating result for TABEL I"
echo " >>>>> Dataset : usroad  with  EBM"
./build/TC ./data/usroad/edge.facts 0
echo " >>>>> Dataset : usroad  without  EBM"
./build/TC ./data/usroad/edge.facts 1
echo " >>>>> "
echo " >>>>> Dataset : vsp_finan  with  EBM"
./build/TC ./data/vsp_finan/edge.facts 0
echo " >>>>> Dataset : vsp_finan  without  EBM"
./build/TC ./data/vsp_finan/edge.facts 1
echo " >>>>> "
echo " >>>>> Dataset : fc_ocean  with  EBM"
./build/TC ./data/fc_ocean/edge.facts 0
echo " >>>>> Dataset : fc_ocean  without  EBM"
./build/TC ./data/fc_ocean/edge.facts 1
echo " >>>>> Dataset : com-dblp  with  EBM"
./build/TC ./data/com-dblp/edge.facts 0
echo " >>>>> Dataset : com-dblp  without  EBM"
./build/TC ./data/com-dblp/edge.facts 1
echo " >>>>> "
echo " >>>>> Dataset : Gnutella31  with  EBM"
./build/TC ./data/Gnutella31/edge.facts 0
echo " >>>>> Dataset : Gnutella31  without  EBM"
./build/TC ./data/Gnutella31/edge.facts 1
echo " >>>>> "

echo " >>>>> Testing GDlog: "
echo " >>>>> Generating result for TABEL II"
# echo "Dataset : usroad"
# ./build/TC ./data/data_165435.txt
echo "Dataset : fc_ocean"
./build/TC ./data/fc_ocean/edge.facts 0
echo "Dataset : com-dblp"
./build/TC ./data/com-dblp/edge.facts 0
echo "Dataset : vsp_finan"
./build/TC ./data/vsp_finan/edge.facts 0
echo "Dataset : Gnutella31"
./build/TC ./data/Gnutella31/edge.facts 0
echo "Dataset : fe_body"
./build/TC ./data/fe_body/edge.facts 0
echo "Dataset : SF.cedge"
./build/TC ./data/SF.cedge/edge.facts 0

