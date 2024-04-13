echo "Preparing to run SG"
git stash && git checkout hash_diff
cmake -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -Bbuild .  && cd build && make -j
cd ..
echo ">>>>>>>>>>>>>>>>>> Testing CSPA >>>>>>>>>>>>>>>>>"
echo " >>>>> Testing GDlog: "
echo "Generating result for TABEL IV"
echo ">>>>>> Dataset : httpd"
./build/CSPA ./data/cspa/httpd
echo ">>>>>> Dataset : linux"
./build/CSPA ./data/cspa/linux
echo ">>>>>> Dataset : postgresql"
./build/CSPA ./data/cspa/postgresql

