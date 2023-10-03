#define SOUFFLE_GENERATOR_VERSION "2.4-13-g63d6684fe"
#include "souffle/CompiledSouffle.h"
#include "souffle/SignalHandler.h"
#include "souffle/SouffleInterface.h"
#include "souffle/datastructure/BTree.h"
#include "souffle/io/IOSystem.h"
#include <any>
namespace functors {
extern "C" {
}
} //namespace functors
namespace souffle::t_btree_ii__0_1__11 {
using namespace souffle;
struct Type {
static constexpr Relation::arity_type Arity = 2;
using t_tuple = Tuple<RamDomain, 2>;
struct t_comparator_0{
 int operator()(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ? -1 : (ramBitCast<RamSigned>(a[0]) > ramBitCast<RamSigned>(b[0])) ? 1 :((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1])) ? -1 : (ramBitCast<RamSigned>(a[1]) > ramBitCast<RamSigned>(b[1])) ? 1 :(0));
 }
bool less(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0]))|| ((ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0])) && ((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1]))));
 }
bool equal(const t_tuple& a, const t_tuple& b) const {
return (ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0]))&&(ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1]));
 }
};
using t_ind_0 = btree_set<t_tuple,t_comparator_0>;
t_ind_0 ind_0;
using iterator = t_ind_0::iterator;
struct context {
t_ind_0::operation_hints hints_0_lower;
t_ind_0::operation_hints hints_0_upper;
};
context createContext() { return context(); }
bool insert(const t_tuple& t);
bool insert(const t_tuple& t, context& h);
bool insert(const RamDomain* ramDomain);
bool insert(RamDomain a0,RamDomain a1);
bool contains(const t_tuple& t, context& h) const;
bool contains(const t_tuple& t) const;
std::size_t size() const;
iterator find(const t_tuple& t, context& h) const;
iterator find(const t_tuple& t) const;
range<iterator> lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const;
range<iterator> lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */) const;
range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper) const;
bool empty() const;
std::vector<range<iterator>> partition() const;
void purge();
iterator begin() const;
iterator end() const;
void printStatistics(std::ostream& o) const;
};
} // namespace souffle::t_btree_ii__0_1__11 
namespace souffle::t_btree_ii__0_1__11 {
using namespace souffle;
using t_ind_0 = Type::t_ind_0;
using iterator = Type::iterator;
using context = Type::context;
bool Type::insert(const t_tuple& t) {
context h;
return insert(t, h);
}
bool Type::insert(const t_tuple& t, context& h) {
if (ind_0.insert(t, h.hints_0_lower)) {
return true;
} else return false;
}
bool Type::insert(const RamDomain* ramDomain) {
RamDomain data[2];
std::copy(ramDomain, ramDomain + 2, data);
const t_tuple& tuple = reinterpret_cast<const t_tuple&>(data);
context h;
return insert(tuple, h);
}
bool Type::insert(RamDomain a0,RamDomain a1) {
RamDomain data[2] = {a0,a1};
return insert(data);
}
bool Type::contains(const t_tuple& t, context& h) const {
return ind_0.contains(t, h.hints_0_lower);
}
bool Type::contains(const t_tuple& t) const {
context h;
return contains(t, h);
}
std::size_t Type::size() const {
return ind_0.size();
}
iterator Type::find(const t_tuple& t, context& h) const {
return ind_0.find(t, h.hints_0_lower);
}
iterator Type::find(const t_tuple& t) const {
context h;
return find(t, h);
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp == 0) {
    auto pos = ind_0.find(lower, h.hints_0_lower);
    auto fin = ind_0.end();
    if (pos != fin) {fin = pos; ++fin;}
    return make_range(pos, fin);
}
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_11(lower,upper,h);
}
bool Type::empty() const {
return ind_0.empty();
}
std::vector<range<iterator>> Type::partition() const {
return ind_0.getChunks(400);
}
void Type::purge() {
ind_0.clear();
}
iterator Type::begin() const {
return ind_0.begin();
}
iterator Type::end() const {
return ind_0.end();
}
void Type::printStatistics(std::ostream& o) const {
o << " arity 2 direct b-tree index 0 lex-order [0,1]\n";
ind_0.printStats(o);
}
} // namespace souffle::t_btree_ii__0_1__11 
namespace souffle::t_btree_ii__0_1__11__10 {
using namespace souffle;
struct Type {
static constexpr Relation::arity_type Arity = 2;
using t_tuple = Tuple<RamDomain, 2>;
struct t_comparator_0{
 int operator()(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ? -1 : (ramBitCast<RamSigned>(a[0]) > ramBitCast<RamSigned>(b[0])) ? 1 :((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1])) ? -1 : (ramBitCast<RamSigned>(a[1]) > ramBitCast<RamSigned>(b[1])) ? 1 :(0));
 }
bool less(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0]))|| ((ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0])) && ((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1]))));
 }
bool equal(const t_tuple& a, const t_tuple& b) const {
return (ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0]))&&(ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1]));
 }
};
using t_ind_0 = btree_set<t_tuple,t_comparator_0>;
t_ind_0 ind_0;
using iterator = t_ind_0::iterator;
struct context {
t_ind_0::operation_hints hints_0_lower;
t_ind_0::operation_hints hints_0_upper;
};
context createContext() { return context(); }
bool insert(const t_tuple& t);
bool insert(const t_tuple& t, context& h);
bool insert(const RamDomain* ramDomain);
bool insert(RamDomain a0,RamDomain a1);
bool contains(const t_tuple& t, context& h) const;
bool contains(const t_tuple& t) const;
std::size_t size() const;
iterator find(const t_tuple& t, context& h) const;
iterator find(const t_tuple& t) const;
range<iterator> lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const;
range<iterator> lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */) const;
range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper) const;
range<t_ind_0::iterator> lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper) const;
bool empty() const;
std::vector<range<iterator>> partition() const;
void purge();
iterator begin() const;
iterator end() const;
void printStatistics(std::ostream& o) const;
};
} // namespace souffle::t_btree_ii__0_1__11__10 
namespace souffle::t_btree_ii__0_1__11__10 {
using namespace souffle;
using t_ind_0 = Type::t_ind_0;
using iterator = Type::iterator;
using context = Type::context;
bool Type::insert(const t_tuple& t) {
context h;
return insert(t, h);
}
bool Type::insert(const t_tuple& t, context& h) {
if (ind_0.insert(t, h.hints_0_lower)) {
return true;
} else return false;
}
bool Type::insert(const RamDomain* ramDomain) {
RamDomain data[2];
std::copy(ramDomain, ramDomain + 2, data);
const t_tuple& tuple = reinterpret_cast<const t_tuple&>(data);
context h;
return insert(tuple, h);
}
bool Type::insert(RamDomain a0,RamDomain a1) {
RamDomain data[2] = {a0,a1};
return insert(data);
}
bool Type::contains(const t_tuple& t, context& h) const {
return ind_0.contains(t, h.hints_0_lower);
}
bool Type::contains(const t_tuple& t) const {
context h;
return contains(t, h);
}
std::size_t Type::size() const {
return ind_0.size();
}
iterator Type::find(const t_tuple& t, context& h) const {
return ind_0.find(t, h.hints_0_lower);
}
iterator Type::find(const t_tuple& t) const {
context h;
return find(t, h);
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp == 0) {
    auto pos = ind_0.find(lower, h.hints_0_lower);
    auto fin = ind_0.end();
    if (pos != fin) {fin = pos; ++fin;}
    return make_range(pos, fin);
}
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_11(lower,upper,h);
}
range<t_ind_0::iterator> Type::lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_10(lower,upper,h);
}
bool Type::empty() const {
return ind_0.empty();
}
std::vector<range<iterator>> Type::partition() const {
return ind_0.getChunks(400);
}
void Type::purge() {
ind_0.clear();
}
iterator Type::begin() const {
return ind_0.begin();
}
iterator Type::end() const {
return ind_0.end();
}
void Type::printStatistics(std::ostream& o) const {
o << " arity 2 direct b-tree index 0 lex-order [0,1]\n";
ind_0.printStats(o);
}
} // namespace souffle::t_btree_ii__0_1__11__10 
namespace  souffle {
using namespace souffle;
class Stratum_MemoryAlias_eb2b99d17f3353cf {
public:
 Stratum_MemoryAlias_eb2b99d17f3353cf(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_delta_MemoryAlias_a006b6c07c4d7f8c,t_btree_ii__0_1__11__10::Type& rel_delta_ValueAlias_ec67472747f9c188,t_btree_ii__0_1__11__10::Type& rel_delta_ValueFlow_66e5c3d04e674156,t_btree_ii__0_1__11__10::Type& rel_new_MemoryAlias_2c2c590c5354b4cd,t_btree_ii__0_1__11__10::Type& rel_new_ValueAlias_3143e2bd0d62924a,t_btree_ii__0_1__11__10::Type& rel_new_ValueFlow_5e353ff29d46b279,t_btree_ii__0_1__11__10::Type& rel_MemoryAlias_9a8750c76c041544,t_btree_ii__0_1__11::Type& rel_ValueAlias_37462775fdb0331f,t_btree_ii__0_1__11__10::Type& rel_ValueFlow_13166cf449369285,t_btree_ii__0_1__11::Type& rel_assign_e4bb6e0824a16a37,t_btree_ii__0_1__11__10::Type& rel_dereference_df51f9c2118ef9bd);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11__10::Type* rel_delta_MemoryAlias_a006b6c07c4d7f8c;
t_btree_ii__0_1__11__10::Type* rel_delta_ValueAlias_ec67472747f9c188;
t_btree_ii__0_1__11__10::Type* rel_delta_ValueFlow_66e5c3d04e674156;
t_btree_ii__0_1__11__10::Type* rel_new_MemoryAlias_2c2c590c5354b4cd;
t_btree_ii__0_1__11__10::Type* rel_new_ValueAlias_3143e2bd0d62924a;
t_btree_ii__0_1__11__10::Type* rel_new_ValueFlow_5e353ff29d46b279;
t_btree_ii__0_1__11__10::Type* rel_MemoryAlias_9a8750c76c041544;
t_btree_ii__0_1__11::Type* rel_ValueAlias_37462775fdb0331f;
t_btree_ii__0_1__11__10::Type* rel_ValueFlow_13166cf449369285;
t_btree_ii__0_1__11::Type* rel_assign_e4bb6e0824a16a37;
t_btree_ii__0_1__11__10::Type* rel_dereference_df51f9c2118ef9bd;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_MemoryAlias_eb2b99d17f3353cf::Stratum_MemoryAlias_eb2b99d17f3353cf(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_delta_MemoryAlias_a006b6c07c4d7f8c,t_btree_ii__0_1__11__10::Type& rel_delta_ValueAlias_ec67472747f9c188,t_btree_ii__0_1__11__10::Type& rel_delta_ValueFlow_66e5c3d04e674156,t_btree_ii__0_1__11__10::Type& rel_new_MemoryAlias_2c2c590c5354b4cd,t_btree_ii__0_1__11__10::Type& rel_new_ValueAlias_3143e2bd0d62924a,t_btree_ii__0_1__11__10::Type& rel_new_ValueFlow_5e353ff29d46b279,t_btree_ii__0_1__11__10::Type& rel_MemoryAlias_9a8750c76c041544,t_btree_ii__0_1__11::Type& rel_ValueAlias_37462775fdb0331f,t_btree_ii__0_1__11__10::Type& rel_ValueFlow_13166cf449369285,t_btree_ii__0_1__11::Type& rel_assign_e4bb6e0824a16a37,t_btree_ii__0_1__11__10::Type& rel_dereference_df51f9c2118ef9bd):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_delta_MemoryAlias_a006b6c07c4d7f8c(&rel_delta_MemoryAlias_a006b6c07c4d7f8c),
rel_delta_ValueAlias_ec67472747f9c188(&rel_delta_ValueAlias_ec67472747f9c188),
rel_delta_ValueFlow_66e5c3d04e674156(&rel_delta_ValueFlow_66e5c3d04e674156),
rel_new_MemoryAlias_2c2c590c5354b4cd(&rel_new_MemoryAlias_2c2c590c5354b4cd),
rel_new_ValueAlias_3143e2bd0d62924a(&rel_new_ValueAlias_3143e2bd0d62924a),
rel_new_ValueFlow_5e353ff29d46b279(&rel_new_ValueFlow_5e353ff29d46b279),
rel_MemoryAlias_9a8750c76c041544(&rel_MemoryAlias_9a8750c76c041544),
rel_ValueAlias_37462775fdb0331f(&rel_ValueAlias_37462775fdb0331f),
rel_ValueFlow_13166cf449369285(&rel_ValueFlow_13166cf449369285),
rel_assign_e4bb6e0824a16a37(&rel_assign_e4bb6e0824a16a37),
rel_dereference_df51f9c2118ef9bd(&rel_dereference_df51f9c2118ef9bd){
}

void Stratum_MemoryAlias_eb2b99d17f3353cf::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(MemoryAlias(x,x) :- 
   assign(_,x).
in file cspa.dl [38:1-38:35])_");
if(!(rel_assign_e4bb6e0824a16a37->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_MemoryAlias_9a8750c76c041544_op_ctxt,rel_MemoryAlias_9a8750c76c041544->createContext());
CREATE_OP_CONTEXT(rel_assign_e4bb6e0824a16a37_op_ctxt,rel_assign_e4bb6e0824a16a37->createContext());
for(const auto& env0 : *rel_assign_e4bb6e0824a16a37) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[1]),ramBitCast(env0[1])}};
rel_MemoryAlias_9a8750c76c041544->insert(tuple,READ_OP_CONTEXT(rel_MemoryAlias_9a8750c76c041544_op_ctxt));
}
}
();}
signalHandler->setMsg(R"_(MemoryAlias(x,x) :- 
   assign(x,_).
in file cspa.dl [39:1-39:35])_");
if(!(rel_assign_e4bb6e0824a16a37->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_MemoryAlias_9a8750c76c041544_op_ctxt,rel_MemoryAlias_9a8750c76c041544->createContext());
CREATE_OP_CONTEXT(rel_assign_e4bb6e0824a16a37_op_ctxt,rel_assign_e4bb6e0824a16a37->createContext());
for(const auto& env0 : *rel_assign_e4bb6e0824a16a37) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[0])}};
rel_MemoryAlias_9a8750c76c041544->insert(tuple,READ_OP_CONTEXT(rel_MemoryAlias_9a8750c76c041544_op_ctxt));
}
}
();}
signalHandler->setMsg(R"_(ValueFlow(y,x) :- 
   assign(y,x).
in file cspa.dl [34:1-34:33])_");
if(!(rel_assign_e4bb6e0824a16a37->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt,rel_ValueFlow_13166cf449369285->createContext());
CREATE_OP_CONTEXT(rel_assign_e4bb6e0824a16a37_op_ctxt,rel_assign_e4bb6e0824a16a37->createContext());
for(const auto& env0 : *rel_assign_e4bb6e0824a16a37) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_ValueFlow_13166cf449369285->insert(tuple,READ_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt));
}
}
();}
signalHandler->setMsg(R"_(ValueFlow(x,x) :- 
   assign(x,_).
in file cspa.dl [35:1-35:33])_");
if(!(rel_assign_e4bb6e0824a16a37->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt,rel_ValueFlow_13166cf449369285->createContext());
CREATE_OP_CONTEXT(rel_assign_e4bb6e0824a16a37_op_ctxt,rel_assign_e4bb6e0824a16a37->createContext());
for(const auto& env0 : *rel_assign_e4bb6e0824a16a37) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[0])}};
rel_ValueFlow_13166cf449369285->insert(tuple,READ_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt));
}
}
();}
signalHandler->setMsg(R"_(ValueFlow(x,x) :- 
   assign(_,x).
in file cspa.dl [36:1-36:33])_");
if(!(rel_assign_e4bb6e0824a16a37->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt,rel_ValueFlow_13166cf449369285->createContext());
CREATE_OP_CONTEXT(rel_assign_e4bb6e0824a16a37_op_ctxt,rel_assign_e4bb6e0824a16a37->createContext());
for(const auto& env0 : *rel_assign_e4bb6e0824a16a37) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[1]),ramBitCast(env0[1])}};
rel_ValueFlow_13166cf449369285->insert(tuple,READ_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt));
}
}
();}
[&](){
CREATE_OP_CONTEXT(rel_delta_MemoryAlias_a006b6c07c4d7f8c_op_ctxt,rel_delta_MemoryAlias_a006b6c07c4d7f8c->createContext());
CREATE_OP_CONTEXT(rel_MemoryAlias_9a8750c76c041544_op_ctxt,rel_MemoryAlias_9a8750c76c041544->createContext());
for(const auto& env0 : *rel_MemoryAlias_9a8750c76c041544) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_MemoryAlias_a006b6c07c4d7f8c->insert(tuple,READ_OP_CONTEXT(rel_delta_MemoryAlias_a006b6c07c4d7f8c_op_ctxt));
}
}
();[&](){
CREATE_OP_CONTEXT(rel_delta_ValueAlias_ec67472747f9c188_op_ctxt,rel_delta_ValueAlias_ec67472747f9c188->createContext());
CREATE_OP_CONTEXT(rel_ValueAlias_37462775fdb0331f_op_ctxt,rel_ValueAlias_37462775fdb0331f->createContext());
for(const auto& env0 : *rel_ValueAlias_37462775fdb0331f) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_ValueAlias_ec67472747f9c188->insert(tuple,READ_OP_CONTEXT(rel_delta_ValueAlias_ec67472747f9c188_op_ctxt));
}
}
();[&](){
CREATE_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt,rel_delta_ValueFlow_66e5c3d04e674156->createContext());
CREATE_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt,rel_ValueFlow_13166cf449369285->createContext());
for(const auto& env0 : *rel_ValueFlow_13166cf449369285) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_ValueFlow_66e5c3d04e674156->insert(tuple,READ_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt));
}
}
();auto loop_counter = RamUnsigned(1);
iter = 0;
for(;;) {
signalHandler->setMsg(R"_(MemoryAlias(x,w) :- 
   dereference(y,x),
   ValueAlias(y,z),
   dereference(z,w).
in file cspa.dl [29:1-29:77])_");
if(!(rel_dereference_df51f9c2118ef9bd->empty()) && !(rel_delta_ValueAlias_ec67472747f9c188->empty())) {
[&](){
auto part = rel_dereference_df51f9c2118ef9bd->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_ValueAlias_ec67472747f9c188_op_ctxt,rel_delta_ValueAlias_ec67472747f9c188->createContext());
CREATE_OP_CONTEXT(rel_new_MemoryAlias_2c2c590c5354b4cd_op_ctxt,rel_new_MemoryAlias_2c2c590c5354b4cd->createContext());
CREATE_OP_CONTEXT(rel_MemoryAlias_9a8750c76c041544_op_ctxt,rel_MemoryAlias_9a8750c76c041544->createContext());
CREATE_OP_CONTEXT(rel_dereference_df51f9c2118ef9bd_op_ctxt,rel_dereference_df51f9c2118ef9bd->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_delta_ValueAlias_ec67472747f9c188->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_delta_ValueAlias_ec67472747f9c188_op_ctxt));
for(const auto& env1 : range) {
auto range = rel_dereference_df51f9c2118ef9bd->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env1[1]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env1[1]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_dereference_df51f9c2118ef9bd_op_ctxt));
for(const auto& env2 : range) {
if( !(rel_MemoryAlias_9a8750c76c041544->contains(Tuple<RamDomain,2>{{ramBitCast(env0[1]),ramBitCast(env2[1])}},READ_OP_CONTEXT(rel_MemoryAlias_9a8750c76c041544_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[1]),ramBitCast(env2[1])}};
rel_new_MemoryAlias_2c2c590c5354b4cd->insert(tuple,READ_OP_CONTEXT(rel_new_MemoryAlias_2c2c590c5354b4cd_op_ctxt));
}
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
signalHandler->setMsg(R"_(ValueAlias(x,y) :- 
   ValueFlow(z,x),
   ValueFlow(z,y).
in file cspa.dl [26:1-26:54])_");
if(!(rel_delta_ValueFlow_66e5c3d04e674156->empty()) && !(rel_ValueFlow_13166cf449369285->empty())) {
[&](){
auto part = rel_delta_ValueFlow_66e5c3d04e674156->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt,rel_delta_ValueFlow_66e5c3d04e674156->createContext());
CREATE_OP_CONTEXT(rel_new_ValueAlias_3143e2bd0d62924a_op_ctxt,rel_new_ValueAlias_3143e2bd0d62924a->createContext());
CREATE_OP_CONTEXT(rel_ValueAlias_37462775fdb0331f_op_ctxt,rel_ValueAlias_37462775fdb0331f->createContext());
CREATE_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt,rel_ValueFlow_13166cf449369285->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_ValueFlow_13166cf449369285->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_ValueAlias_37462775fdb0331f->contains(Tuple<RamDomain,2>{{ramBitCast(env0[1]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_ValueAlias_37462775fdb0331f_op_ctxt))) && !(rel_delta_ValueFlow_66e5c3d04e674156->contains(Tuple<RamDomain,2>{{ramBitCast(env0[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[1]),ramBitCast(env1[1])}};
rel_new_ValueAlias_3143e2bd0d62924a->insert(tuple,READ_OP_CONTEXT(rel_new_ValueAlias_3143e2bd0d62924a_op_ctxt));
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
signalHandler->setMsg(R"_(ValueAlias(x,y) :- 
   ValueFlow(z,x),
   ValueFlow(z,y).
in file cspa.dl [26:1-26:54])_");
if(!(rel_ValueFlow_13166cf449369285->empty()) && !(rel_delta_ValueFlow_66e5c3d04e674156->empty())) {
[&](){
auto part = rel_ValueFlow_13166cf449369285->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt,rel_delta_ValueFlow_66e5c3d04e674156->createContext());
CREATE_OP_CONTEXT(rel_new_ValueAlias_3143e2bd0d62924a_op_ctxt,rel_new_ValueAlias_3143e2bd0d62924a->createContext());
CREATE_OP_CONTEXT(rel_ValueAlias_37462775fdb0331f_op_ctxt,rel_ValueAlias_37462775fdb0331f->createContext());
CREATE_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt,rel_ValueFlow_13166cf449369285->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_delta_ValueFlow_66e5c3d04e674156->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_ValueAlias_37462775fdb0331f->contains(Tuple<RamDomain,2>{{ramBitCast(env0[1]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_ValueAlias_37462775fdb0331f_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[1]),ramBitCast(env1[1])}};
rel_new_ValueAlias_3143e2bd0d62924a->insert(tuple,READ_OP_CONTEXT(rel_new_ValueAlias_3143e2bd0d62924a_op_ctxt));
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
signalHandler->setMsg(R"_(ValueAlias(x,y) :- 
   ValueFlow(z,x),
   MemoryAlias(z,w),
   ValueFlow(w,y).
in file cspa.dl [31:1-31:73])_");
if(!(rel_MemoryAlias_9a8750c76c041544->empty()) && !(rel_ValueFlow_13166cf449369285->empty()) && !(rel_delta_ValueFlow_66e5c3d04e674156->empty())) {
[&](){
auto part = rel_delta_ValueFlow_66e5c3d04e674156->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_MemoryAlias_a006b6c07c4d7f8c_op_ctxt,rel_delta_MemoryAlias_a006b6c07c4d7f8c->createContext());
CREATE_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt,rel_delta_ValueFlow_66e5c3d04e674156->createContext());
CREATE_OP_CONTEXT(rel_new_ValueAlias_3143e2bd0d62924a_op_ctxt,rel_new_ValueAlias_3143e2bd0d62924a->createContext());
CREATE_OP_CONTEXT(rel_MemoryAlias_9a8750c76c041544_op_ctxt,rel_MemoryAlias_9a8750c76c041544->createContext());
CREATE_OP_CONTEXT(rel_ValueAlias_37462775fdb0331f_op_ctxt,rel_ValueAlias_37462775fdb0331f->createContext());
CREATE_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt,rel_ValueFlow_13166cf449369285->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_MemoryAlias_9a8750c76c041544->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_MemoryAlias_9a8750c76c041544_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_delta_MemoryAlias_a006b6c07c4d7f8c->contains(Tuple<RamDomain,2>{{ramBitCast(env0[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_delta_MemoryAlias_a006b6c07c4d7f8c_op_ctxt)))) {
auto range = rel_ValueFlow_13166cf449369285->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env1[1]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env1[1]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt));
for(const auto& env2 : range) {
if( !(rel_ValueAlias_37462775fdb0331f->contains(Tuple<RamDomain,2>{{ramBitCast(env0[1]),ramBitCast(env2[1])}},READ_OP_CONTEXT(rel_ValueAlias_37462775fdb0331f_op_ctxt))) && !(rel_delta_ValueFlow_66e5c3d04e674156->contains(Tuple<RamDomain,2>{{ramBitCast(env1[1]),ramBitCast(env2[1])}},READ_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[1]),ramBitCast(env2[1])}};
rel_new_ValueAlias_3143e2bd0d62924a->insert(tuple,READ_OP_CONTEXT(rel_new_ValueAlias_3143e2bd0d62924a_op_ctxt));
}
}
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
signalHandler->setMsg(R"_(ValueAlias(x,y) :- 
   ValueFlow(z,x),
   MemoryAlias(z,w),
   ValueFlow(w,y).
in file cspa.dl [31:1-31:73])_");
if(!(rel_ValueFlow_13166cf449369285->empty()) && !(rel_delta_MemoryAlias_a006b6c07c4d7f8c->empty())) {
[&](){
auto part = rel_ValueFlow_13166cf449369285->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_MemoryAlias_a006b6c07c4d7f8c_op_ctxt,rel_delta_MemoryAlias_a006b6c07c4d7f8c->createContext());
CREATE_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt,rel_delta_ValueFlow_66e5c3d04e674156->createContext());
CREATE_OP_CONTEXT(rel_new_ValueAlias_3143e2bd0d62924a_op_ctxt,rel_new_ValueAlias_3143e2bd0d62924a->createContext());
CREATE_OP_CONTEXT(rel_ValueAlias_37462775fdb0331f_op_ctxt,rel_ValueAlias_37462775fdb0331f->createContext());
CREATE_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt,rel_ValueFlow_13166cf449369285->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_delta_MemoryAlias_a006b6c07c4d7f8c->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_delta_MemoryAlias_a006b6c07c4d7f8c_op_ctxt));
for(const auto& env1 : range) {
auto range = rel_ValueFlow_13166cf449369285->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env1[1]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env1[1]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt));
for(const auto& env2 : range) {
if( !(rel_ValueAlias_37462775fdb0331f->contains(Tuple<RamDomain,2>{{ramBitCast(env0[1]),ramBitCast(env2[1])}},READ_OP_CONTEXT(rel_ValueAlias_37462775fdb0331f_op_ctxt))) && !(rel_delta_ValueFlow_66e5c3d04e674156->contains(Tuple<RamDomain,2>{{ramBitCast(env1[1]),ramBitCast(env2[1])}},READ_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[1]),ramBitCast(env2[1])}};
rel_new_ValueAlias_3143e2bd0d62924a->insert(tuple,READ_OP_CONTEXT(rel_new_ValueAlias_3143e2bd0d62924a_op_ctxt));
}
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
signalHandler->setMsg(R"_(ValueAlias(x,y) :- 
   ValueFlow(z,x),
   MemoryAlias(z,w),
   ValueFlow(w,y).
in file cspa.dl [31:1-31:73])_");
if(!(rel_MemoryAlias_9a8750c76c041544->empty()) && !(rel_delta_ValueFlow_66e5c3d04e674156->empty()) && !(rel_ValueFlow_13166cf449369285->empty())) {
[&](){
auto part = rel_ValueFlow_13166cf449369285->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt,rel_delta_ValueFlow_66e5c3d04e674156->createContext());
CREATE_OP_CONTEXT(rel_new_ValueAlias_3143e2bd0d62924a_op_ctxt,rel_new_ValueAlias_3143e2bd0d62924a->createContext());
CREATE_OP_CONTEXT(rel_MemoryAlias_9a8750c76c041544_op_ctxt,rel_MemoryAlias_9a8750c76c041544->createContext());
CREATE_OP_CONTEXT(rel_ValueAlias_37462775fdb0331f_op_ctxt,rel_ValueAlias_37462775fdb0331f->createContext());
CREATE_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt,rel_ValueFlow_13166cf449369285->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_MemoryAlias_9a8750c76c041544->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_MemoryAlias_9a8750c76c041544_op_ctxt));
for(const auto& env1 : range) {
auto range = rel_delta_ValueFlow_66e5c3d04e674156->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env1[1]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env1[1]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt));
for(const auto& env2 : range) {
if( !(rel_ValueAlias_37462775fdb0331f->contains(Tuple<RamDomain,2>{{ramBitCast(env0[1]),ramBitCast(env2[1])}},READ_OP_CONTEXT(rel_ValueAlias_37462775fdb0331f_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[1]),ramBitCast(env2[1])}};
rel_new_ValueAlias_3143e2bd0d62924a->insert(tuple,READ_OP_CONTEXT(rel_new_ValueAlias_3143e2bd0d62924a_op_ctxt));
}
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
signalHandler->setMsg(R"_(ValueFlow(x,y) :- 
   ValueFlow(x,z),
   ValueFlow(z,y).
in file cspa.dl [25:1-25:53])_");
if(!(rel_delta_ValueFlow_66e5c3d04e674156->empty()) && !(rel_ValueFlow_13166cf449369285->empty())) {
[&](){
auto part = rel_delta_ValueFlow_66e5c3d04e674156->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt,rel_delta_ValueFlow_66e5c3d04e674156->createContext());
CREATE_OP_CONTEXT(rel_new_ValueFlow_5e353ff29d46b279_op_ctxt,rel_new_ValueFlow_5e353ff29d46b279->createContext());
CREATE_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt,rel_ValueFlow_13166cf449369285->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_ValueFlow_13166cf449369285->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[1]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[1]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_ValueFlow_13166cf449369285->contains(Tuple<RamDomain,2>{{ramBitCast(env0[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt))) && !(rel_delta_ValueFlow_66e5c3d04e674156->contains(Tuple<RamDomain,2>{{ramBitCast(env0[1]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env1[1])}};
rel_new_ValueFlow_5e353ff29d46b279->insert(tuple,READ_OP_CONTEXT(rel_new_ValueFlow_5e353ff29d46b279_op_ctxt));
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
signalHandler->setMsg(R"_(ValueFlow(x,y) :- 
   ValueFlow(x,z),
   ValueFlow(z,y).
in file cspa.dl [25:1-25:53])_");
if(!(rel_ValueFlow_13166cf449369285->empty()) && !(rel_delta_ValueFlow_66e5c3d04e674156->empty())) {
[&](){
auto part = rel_ValueFlow_13166cf449369285->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt,rel_delta_ValueFlow_66e5c3d04e674156->createContext());
CREATE_OP_CONTEXT(rel_new_ValueFlow_5e353ff29d46b279_op_ctxt,rel_new_ValueFlow_5e353ff29d46b279->createContext());
CREATE_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt,rel_ValueFlow_13166cf449369285->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_delta_ValueFlow_66e5c3d04e674156->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[1]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[1]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_delta_ValueFlow_66e5c3d04e674156_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_ValueFlow_13166cf449369285->contains(Tuple<RamDomain,2>{{ramBitCast(env0[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env1[1])}};
rel_new_ValueFlow_5e353ff29d46b279->insert(tuple,READ_OP_CONTEXT(rel_new_ValueFlow_5e353ff29d46b279_op_ctxt));
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
signalHandler->setMsg(R"_(ValueFlow(x,y) :- 
   assign(x,z),
   MemoryAlias(z,y).
in file cspa.dl [27:1-27:52])_");
if(!(rel_assign_e4bb6e0824a16a37->empty()) && !(rel_delta_MemoryAlias_a006b6c07c4d7f8c->empty())) {
[&](){
auto part = rel_assign_e4bb6e0824a16a37->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_MemoryAlias_a006b6c07c4d7f8c_op_ctxt,rel_delta_MemoryAlias_a006b6c07c4d7f8c->createContext());
CREATE_OP_CONTEXT(rel_new_ValueFlow_5e353ff29d46b279_op_ctxt,rel_new_ValueFlow_5e353ff29d46b279->createContext());
CREATE_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt,rel_ValueFlow_13166cf449369285->createContext());
CREATE_OP_CONTEXT(rel_assign_e4bb6e0824a16a37_op_ctxt,rel_assign_e4bb6e0824a16a37->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_delta_MemoryAlias_a006b6c07c4d7f8c->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[1]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[1]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_delta_MemoryAlias_a006b6c07c4d7f8c_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_ValueFlow_13166cf449369285->contains(Tuple<RamDomain,2>{{ramBitCast(env0[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env1[1])}};
rel_new_ValueFlow_5e353ff29d46b279->insert(tuple,READ_OP_CONTEXT(rel_new_ValueFlow_5e353ff29d46b279_op_ctxt));
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
if(rel_new_MemoryAlias_2c2c590c5354b4cd->empty() && rel_new_ValueAlias_3143e2bd0d62924a->empty() && rel_new_ValueFlow_5e353ff29d46b279->empty()) break;
[&](){
CREATE_OP_CONTEXT(rel_new_MemoryAlias_2c2c590c5354b4cd_op_ctxt,rel_new_MemoryAlias_2c2c590c5354b4cd->createContext());
CREATE_OP_CONTEXT(rel_MemoryAlias_9a8750c76c041544_op_ctxt,rel_MemoryAlias_9a8750c76c041544->createContext());
for(const auto& env0 : *rel_new_MemoryAlias_2c2c590c5354b4cd) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_MemoryAlias_9a8750c76c041544->insert(tuple,READ_OP_CONTEXT(rel_MemoryAlias_9a8750c76c041544_op_ctxt));
}
}
();std::swap(rel_delta_MemoryAlias_a006b6c07c4d7f8c, rel_new_MemoryAlias_2c2c590c5354b4cd);
rel_new_MemoryAlias_2c2c590c5354b4cd->purge();
[&](){
CREATE_OP_CONTEXT(rel_new_ValueAlias_3143e2bd0d62924a_op_ctxt,rel_new_ValueAlias_3143e2bd0d62924a->createContext());
CREATE_OP_CONTEXT(rel_ValueAlias_37462775fdb0331f_op_ctxt,rel_ValueAlias_37462775fdb0331f->createContext());
for(const auto& env0 : *rel_new_ValueAlias_3143e2bd0d62924a) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_ValueAlias_37462775fdb0331f->insert(tuple,READ_OP_CONTEXT(rel_ValueAlias_37462775fdb0331f_op_ctxt));
}
}
();std::swap(rel_delta_ValueAlias_ec67472747f9c188, rel_new_ValueAlias_3143e2bd0d62924a);
rel_new_ValueAlias_3143e2bd0d62924a->purge();
[&](){
CREATE_OP_CONTEXT(rel_new_ValueFlow_5e353ff29d46b279_op_ctxt,rel_new_ValueFlow_5e353ff29d46b279->createContext());
CREATE_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt,rel_ValueFlow_13166cf449369285->createContext());
for(const auto& env0 : *rel_new_ValueFlow_5e353ff29d46b279) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_ValueFlow_13166cf449369285->insert(tuple,READ_OP_CONTEXT(rel_ValueFlow_13166cf449369285_op_ctxt));
}
}
();std::swap(rel_delta_ValueFlow_66e5c3d04e674156, rel_new_ValueFlow_5e353ff29d46b279);
rel_new_ValueFlow_5e353ff29d46b279->purge();
loop_counter = (ramBitCast<RamUnsigned>(loop_counter) + ramBitCast<RamUnsigned>(RamUnsigned(1)));
iter++;
}
iter = 0;
rel_delta_MemoryAlias_a006b6c07c4d7f8c->purge();
rel_new_MemoryAlias_2c2c590c5354b4cd->purge();
rel_delta_ValueAlias_ec67472747f9c188->purge();
rel_new_ValueAlias_3143e2bd0d62924a->purge();
rel_delta_ValueFlow_66e5c3d04e674156->purge();
rel_new_ValueFlow_5e353ff29d46b279->purge();
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","src\tdest"},{"auxArity","0"},{"name","MemoryAlias"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_MemoryAlias_9a8750c76c041544);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","src\tdest"},{"auxArity","0"},{"name","ValueAlias"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_ValueAlias_37462775fdb0331f);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","src\tdest"},{"auxArity","0"},{"name","ValueAlias"},{"operation","output"},{"output-dir","."},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_ValueAlias_37462775fdb0331f);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","src\tdest"},{"auxArity","0"},{"name","ValueFlow"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_ValueFlow_13166cf449369285);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","src\tdest"},{"auxArity","0"},{"name","ValueFlow"},{"operation","output"},{"output-dir","."},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_ValueFlow_13166cf449369285);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (pruneImdtRels) rel_assign_e4bb6e0824a16a37->purge();
if (pruneImdtRels) rel_dereference_df51f9c2118ef9bd->purge();
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_assign_e0d78e44f4df6411 {
public:
 Stratum_assign_e0d78e44f4df6411(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_assign_e4bb6e0824a16a37);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_assign_e4bb6e0824a16a37;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_assign_e0d78e44f4df6411::Stratum_assign_e0d78e44f4df6411(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_assign_e4bb6e0824a16a37):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_assign_e4bb6e0824a16a37(&rel_assign_e4bb6e0824a16a37){
}

void Stratum_assign_e0d78e44f4df6411::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","src\tdest"},{"auxArity","0"},{"deliminator","\t"},{"fact-dir","."},{"name","assign"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectory.empty()) {directiveMap["fact-dir"] = inputDirectory;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_assign_e4bb6e0824a16a37);
} catch (std::exception& e) {std::cerr << "Error loading assign data: " << e.what() << '\n';
exit(1);
}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_dereference_9790dfd719959834 {
public:
 Stratum_dereference_9790dfd719959834(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_dereference_df51f9c2118ef9bd);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11__10::Type* rel_dereference_df51f9c2118ef9bd;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_dereference_9790dfd719959834::Stratum_dereference_9790dfd719959834(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_dereference_df51f9c2118ef9bd):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_dereference_df51f9c2118ef9bd(&rel_dereference_df51f9c2118ef9bd){
}

void Stratum_dereference_9790dfd719959834::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","src\tdest"},{"auxArity","0"},{"deliminator","\t"},{"fact-dir","."},{"name","dereference"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectory.empty()) {directiveMap["fact-dir"] = inputDirectory;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_dereference_df51f9c2118ef9bd);
} catch (std::exception& e) {std::cerr << "Error loading dereference data: " << e.what() << '\n';
exit(1);
}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_tmp_ff6f1a280c21bb7c {
public:
 Stratum_tmp_ff6f1a280c21bb7c(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_tmp_033a558889f5439d);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_tmp_033a558889f5439d;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_tmp_ff6f1a280c21bb7c::Stratum_tmp_ff6f1a280c21bb7c(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_tmp_033a558889f5439d):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_tmp_033a558889f5439d(&rel_tmp_033a558889f5439d){
}

void Stratum_tmp_ff6f1a280c21bb7c::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","src\tdest"},{"auxArity","0"},{"name","tmp"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_tmp_033a558889f5439d);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Sf_cspa: public SouffleProgram {
public:
 Sf_cspa();
 ~Sf_cspa();
void run();
void runAll(std::string inputDirectoryArg = "",std::string outputDirectoryArg = "",bool performIOArg = true,bool pruneImdtRelsArg = true);
void printAll([[maybe_unused]] std::string outputDirectoryArg = "");
void loadAll([[maybe_unused]] std::string inputDirectoryArg = "");
void dumpInputs();
void dumpOutputs();
SymbolTable& getSymbolTable();
RecordTable& getRecordTable();
void setNumThreads(std::size_t numThreadsValue);
void executeSubroutine(std::string name,const std::vector<RamDomain>& args,std::vector<RamDomain>& ret);
private:
void runFunction(std::string inputDirectoryArg,std::string outputDirectoryArg,bool performIOArg,bool pruneImdtRelsArg);
SymbolTableImpl symTable;
SpecializedRecordTable<0> recordTable;
ConcurrentCache<std::string,std::regex> regexCache;
Own<t_btree_ii__0_1__11::Type> rel_assign_e4bb6e0824a16a37;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_assign_e4bb6e0824a16a37;
Own<t_btree_ii__0_1__11__10::Type> rel_dereference_df51f9c2118ef9bd;
souffle::RelationWrapper<t_btree_ii__0_1__11__10::Type> wrapper_rel_dereference_df51f9c2118ef9bd;
Own<t_btree_ii__0_1__11__10::Type> rel_MemoryAlias_9a8750c76c041544;
souffle::RelationWrapper<t_btree_ii__0_1__11__10::Type> wrapper_rel_MemoryAlias_9a8750c76c041544;
Own<t_btree_ii__0_1__11__10::Type> rel_delta_MemoryAlias_a006b6c07c4d7f8c;
Own<t_btree_ii__0_1__11__10::Type> rel_new_MemoryAlias_2c2c590c5354b4cd;
Own<t_btree_ii__0_1__11::Type> rel_ValueAlias_37462775fdb0331f;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_ValueAlias_37462775fdb0331f;
Own<t_btree_ii__0_1__11__10::Type> rel_delta_ValueAlias_ec67472747f9c188;
Own<t_btree_ii__0_1__11__10::Type> rel_new_ValueAlias_3143e2bd0d62924a;
Own<t_btree_ii__0_1__11__10::Type> rel_ValueFlow_13166cf449369285;
souffle::RelationWrapper<t_btree_ii__0_1__11__10::Type> wrapper_rel_ValueFlow_13166cf449369285;
Own<t_btree_ii__0_1__11__10::Type> rel_delta_ValueFlow_66e5c3d04e674156;
Own<t_btree_ii__0_1__11__10::Type> rel_new_ValueFlow_5e353ff29d46b279;
Own<t_btree_ii__0_1__11::Type> rel_tmp_033a558889f5439d;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_tmp_033a558889f5439d;
Stratum_MemoryAlias_eb2b99d17f3353cf stratum_MemoryAlias_09e3f01dfdc49ff9;
Stratum_assign_e0d78e44f4df6411 stratum_assign_f550d366a9215d2a;
Stratum_dereference_9790dfd719959834 stratum_dereference_50d427b1fb0ff09b;
Stratum_tmp_ff6f1a280c21bb7c stratum_tmp_88325b656f03e9c6;
std::string inputDirectory;
std::string outputDirectory;
SignalHandler* signalHandler{SignalHandler::instance()};
std::atomic<RamDomain> ctr{};
std::atomic<std::size_t> iter{};
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Sf_cspa::Sf_cspa():
symTable(),
recordTable(),
regexCache(),
rel_assign_e4bb6e0824a16a37(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_assign_e4bb6e0824a16a37(0, *rel_assign_e4bb6e0824a16a37, *this, "assign", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"src","dest"}}, 0),
rel_dereference_df51f9c2118ef9bd(mk<t_btree_ii__0_1__11__10::Type>()),
wrapper_rel_dereference_df51f9c2118ef9bd(1, *rel_dereference_df51f9c2118ef9bd, *this, "dereference", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"src","dest"}}, 0),
rel_MemoryAlias_9a8750c76c041544(mk<t_btree_ii__0_1__11__10::Type>()),
wrapper_rel_MemoryAlias_9a8750c76c041544(2, *rel_MemoryAlias_9a8750c76c041544, *this, "MemoryAlias", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"src","dest"}}, 0),
rel_delta_MemoryAlias_a006b6c07c4d7f8c(mk<t_btree_ii__0_1__11__10::Type>()),
rel_new_MemoryAlias_2c2c590c5354b4cd(mk<t_btree_ii__0_1__11__10::Type>()),
rel_ValueAlias_37462775fdb0331f(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_ValueAlias_37462775fdb0331f(3, *rel_ValueAlias_37462775fdb0331f, *this, "ValueAlias", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"src","dest"}}, 0),
rel_delta_ValueAlias_ec67472747f9c188(mk<t_btree_ii__0_1__11__10::Type>()),
rel_new_ValueAlias_3143e2bd0d62924a(mk<t_btree_ii__0_1__11__10::Type>()),
rel_ValueFlow_13166cf449369285(mk<t_btree_ii__0_1__11__10::Type>()),
wrapper_rel_ValueFlow_13166cf449369285(4, *rel_ValueFlow_13166cf449369285, *this, "ValueFlow", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"src","dest"}}, 0),
rel_delta_ValueFlow_66e5c3d04e674156(mk<t_btree_ii__0_1__11__10::Type>()),
rel_new_ValueFlow_5e353ff29d46b279(mk<t_btree_ii__0_1__11__10::Type>()),
rel_tmp_033a558889f5439d(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_tmp_033a558889f5439d(5, *rel_tmp_033a558889f5439d, *this, "tmp", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"src","dest"}}, 0),
stratum_MemoryAlias_09e3f01dfdc49ff9(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_delta_MemoryAlias_a006b6c07c4d7f8c,*rel_delta_ValueAlias_ec67472747f9c188,*rel_delta_ValueFlow_66e5c3d04e674156,*rel_new_MemoryAlias_2c2c590c5354b4cd,*rel_new_ValueAlias_3143e2bd0d62924a,*rel_new_ValueFlow_5e353ff29d46b279,*rel_MemoryAlias_9a8750c76c041544,*rel_ValueAlias_37462775fdb0331f,*rel_ValueFlow_13166cf449369285,*rel_assign_e4bb6e0824a16a37,*rel_dereference_df51f9c2118ef9bd),
stratum_assign_f550d366a9215d2a(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_assign_e4bb6e0824a16a37),
stratum_dereference_50d427b1fb0ff09b(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_dereference_df51f9c2118ef9bd),
stratum_tmp_88325b656f03e9c6(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_tmp_033a558889f5439d){
addRelation("assign", wrapper_rel_assign_e4bb6e0824a16a37, true, false);
addRelation("dereference", wrapper_rel_dereference_df51f9c2118ef9bd, true, false);
addRelation("MemoryAlias", wrapper_rel_MemoryAlias_9a8750c76c041544, false, true);
addRelation("ValueAlias", wrapper_rel_ValueAlias_37462775fdb0331f, false, true);
addRelation("ValueFlow", wrapper_rel_ValueFlow_13166cf449369285, false, true);
addRelation("tmp", wrapper_rel_tmp_033a558889f5439d, false, true);
}

 Sf_cspa::~Sf_cspa(){
}

void Sf_cspa::runFunction(std::string inputDirectoryArg,std::string outputDirectoryArg,bool performIOArg,bool pruneImdtRelsArg){

    this->inputDirectory  = std::move(inputDirectoryArg);
    this->outputDirectory = std::move(outputDirectoryArg);
    this->performIO       = performIOArg;
    this->pruneImdtRels   = pruneImdtRelsArg;

    // set default threads (in embedded mode)
    // if this is not set, and omp is used, the default omp setting of number of cores is used.
#if defined(_OPENMP)
    if (0 < getNumThreads()) { omp_set_num_threads(static_cast<int>(getNumThreads())); }
#endif

    signalHandler->set();
// -- query evaluation --
{
 std::vector<RamDomain> args, ret;
stratum_assign_f550d366a9215d2a.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_dereference_50d427b1fb0ff09b.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_MemoryAlias_09e3f01dfdc49ff9.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_tmp_88325b656f03e9c6.run(args, ret);
}

// -- relation hint statistics --
signalHandler->reset();
}

void Sf_cspa::run(){
runFunction("", "", false, false);
}

void Sf_cspa::runAll(std::string inputDirectoryArg,std::string outputDirectoryArg,bool performIOArg,bool pruneImdtRelsArg){
runFunction(inputDirectoryArg, outputDirectoryArg, performIOArg, pruneImdtRelsArg);
}

void Sf_cspa::printAll([[maybe_unused]] std::string outputDirectoryArg){
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","src\tdest"},{"auxArity","0"},{"name","MemoryAlias"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_MemoryAlias_9a8750c76c041544);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","src\tdest"},{"auxArity","0"},{"name","ValueAlias"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_ValueAlias_37462775fdb0331f);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","src\tdest"},{"auxArity","0"},{"name","ValueAlias"},{"operation","output"},{"output-dir","."},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_ValueAlias_37462775fdb0331f);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","src\tdest"},{"auxArity","0"},{"name","ValueFlow"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_ValueFlow_13166cf449369285);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","src\tdest"},{"auxArity","0"},{"name","ValueFlow"},{"operation","output"},{"output-dir","."},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_ValueFlow_13166cf449369285);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","src\tdest"},{"auxArity","0"},{"name","tmp"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_tmp_033a558889f5439d);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}

void Sf_cspa::loadAll([[maybe_unused]] std::string inputDirectoryArg){
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","src\tdest"},{"auxArity","0"},{"deliminator","\t"},{"fact-dir","."},{"name","assign"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectoryArg.empty()) {directiveMap["fact-dir"] = inputDirectoryArg;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_assign_e4bb6e0824a16a37);
} catch (std::exception& e) {std::cerr << "Error loading assign data: " << e.what() << '\n';
exit(1);
}
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","src\tdest"},{"auxArity","0"},{"deliminator","\t"},{"fact-dir","."},{"name","dereference"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"src\", \"dest\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectoryArg.empty()) {directiveMap["fact-dir"] = inputDirectoryArg;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_dereference_df51f9c2118ef9bd);
} catch (std::exception& e) {std::cerr << "Error loading dereference data: " << e.what() << '\n';
exit(1);
}
}

void Sf_cspa::dumpInputs(){
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "assign";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_assign_e4bb6e0824a16a37);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "dereference";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_dereference_df51f9c2118ef9bd);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}

void Sf_cspa::dumpOutputs(){
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "MemoryAlias";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_MemoryAlias_9a8750c76c041544);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "ValueAlias";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_ValueAlias_37462775fdb0331f);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "ValueAlias";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_ValueAlias_37462775fdb0331f);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "ValueFlow";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_ValueFlow_13166cf449369285);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "ValueFlow";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_ValueFlow_13166cf449369285);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "tmp";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_tmp_033a558889f5439d);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}

SymbolTable& Sf_cspa::getSymbolTable(){
return symTable;
}

RecordTable& Sf_cspa::getRecordTable(){
return recordTable;
}

void Sf_cspa::setNumThreads(std::size_t numThreadsValue){
SouffleProgram::setNumThreads(numThreadsValue);
symTable.setNumLanes(getNumThreads());
recordTable.setNumLanes(getNumThreads());
regexCache.setNumLanes(getNumThreads());
}

void Sf_cspa::executeSubroutine(std::string name,const std::vector<RamDomain>& args,std::vector<RamDomain>& ret){
if (name == "MemoryAlias") {
stratum_MemoryAlias_09e3f01dfdc49ff9.run(args, ret);
return;}
if (name == "assign") {
stratum_assign_f550d366a9215d2a.run(args, ret);
return;}
if (name == "dereference") {
stratum_dereference_50d427b1fb0ff09b.run(args, ret);
return;}
if (name == "tmp") {
stratum_tmp_88325b656f03e9c6.run(args, ret);
return;}
fatal(("unknown subroutine " + name).c_str());
}

} // namespace  souffle
namespace souffle {
SouffleProgram *newInstance_cspa(){return new  souffle::Sf_cspa;}
SymbolTable *getST_cspa(SouffleProgram *p){return &reinterpret_cast<souffle::Sf_cspa*>(p)->getSymbolTable();}
} // namespace souffle

#ifndef __EMBEDDED_SOUFFLE__
#include "souffle/CompiledOptions.h"
int main(int argc, char** argv)
{
try{
souffle::CmdOptions opt(R"(./cspa.dl)",
R"()",
R"()",
false,
R"()",
32);
if (!opt.parse(argc,argv)) return 1;
souffle::Sf_cspa obj;
#if defined(_OPENMP) 
obj.setNumThreads(opt.getNumJobs());

#endif
obj.runAll(opt.getInputFileDir(), opt.getOutputFileDir());
return 0;
} catch(std::exception &e) { souffle::SignalHandler::instance()->error(e.what());}
}
#endif

namespace  souffle {
using namespace souffle;
class factory_Sf_cspa: souffle::ProgramFactory {
public:
souffle::SouffleProgram* newInstance();
 factory_Sf_cspa();
private:
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
souffle::SouffleProgram* factory_Sf_cspa::newInstance(){
return new  souffle::Sf_cspa();
}

 factory_Sf_cspa::factory_Sf_cspa():
souffle::ProgramFactory("cspa"){
}

} // namespace  souffle
namespace souffle {

#ifdef __EMBEDDED_SOUFFLE__
extern "C" {
souffle::factory_Sf_cspa __factory_Sf_cspa_instance;
}
#endif
} // namespace souffle

