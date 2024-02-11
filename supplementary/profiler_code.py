# SOME CODE FOR USING A PROFILER

#with torch.autograd.profiler.profile(use_cuda=True) as prof:
#CODE
#print(prof.key_averages().table(sort_by="cuda_time_total"))
'''
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")
#alternatively, torch.autograd.profiler.profile(...)
with torch.profiler.profile(use_cuda=True,
                                        schedule=torch.profiler.schedule(skip_first=10,wait=5,warmup=1,active=2),
                                        on_trace_ready=trace_handler) as prof:
# for detailed instructions, see: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
'''