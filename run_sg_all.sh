echo "Preparing code and building SG"
git stash && git checkout hash_diff 
cmake -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -Bbuild . && cd build && make -j 
cd ..
echo ">>>>>>>>>>>>>>>>>> Testing SG >>>>>>>>>>>>>>>>>"
echo " >>>>> Testing GDlog: "
echo "Generating result for TABEL III"
echo "Dataset : fe_body"
./build/SG ./data/fe_body/edge.facts
echo "Dataset : loc-Brightkite"
./build/SG ./data/loc_Brightkite/edge.facts
echo "Dataset : fe-sphere"
./build/SG ./data/fe-sphere/edge.facts
echo "Dataset : CA-HepTH"
./build/SG ./data/CA-HepTH/edge.facts
echo "Dataset : SF.cedge"
./build/SG ./data/SF.cedge/edge.facts
echo "Dataset : ego-Facebook"
./build/SG ./data/ego-Facebook/edge.facts
