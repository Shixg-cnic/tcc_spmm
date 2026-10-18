#./compute_csv_metric_category_div \
#    profiling/41-measure-bcsr-bcsc.csv \
#    "Kernel Name" \
#    l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum \
#    l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum \
#    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
#    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum

./compute_hybrid_l2_cache_hit_rate ${1} \
    "Kernel Name"   \
    lts__t_sectors_srcunit_tex_lookup_hit.sum \
    lts__t_sectors_srcunit_ltcfabric_lookup_hit.sum \
    lts__t_sectors_srcunit_tex_lookup_miss.sum \
    lts__t_sectors_srcunit_ltcfabric_lookup_miss.sum \
    &> /tmp/38000000.txt