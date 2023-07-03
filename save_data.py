
def save_results(res, task_name, time):

    results_test_dir = 'results_test' + '/' + task_name + '/'
    word_name = os.path.exists(results_test_dir)
    if not word_name:
        os.makedirs(results_test_dir)
    time = int(time/60)

    colu = [f'col_{i}' for i in range(4)]   # 用来做列名
    colu1 = ['acc', 'sen', 'spe', 'auc']
    inde = [f'fold_{i}' for i in range(5)]  # 用来做索引
    df1 = pd.DataFrame(data=res, index=inde, columns=colu1)
    df2 = pd.DataFrame({'time':time}, index=[5])

    df = pd.concat([df1,df2], ignore_index=True)
    # task_name = "AD_classification"
    df.to_excel("/media/yingpan/privacy/Z1Ting/results_test/{}/validation_performance.xlsx".format(task_name))





def save_test_results(avg_test_results, task_name):
    results_test_dir = '/media/yingpan/privacy/Z1Ting/'+'results_test' + '/' + task_name + '/'
    word_name = os.path.exists(results_test_dir)
    if not word_name:
        os.makedirs(results_test_dir)

    colu = ['ACC', 'SEN', 'SPE', 'AUC']
    inde = ['test_avg_performance']
    df = pd.DataFrame(data=avg_test_results, index=colu, columns=inde)
    df.to_excel("/media/yingpan/privacy/Z1Ting/results_test/{}/test_avg_performance.xlsx".format(task_name))






def save_patch_result(list, task_name):
    list = list
    df = pd.DataFrame(list)
    df.to_excel("G:/select_patch/AD_classification/patch-acc-results.xlsx")

list = [1,1,1,1]
task_name = 'AD_classification'

save_patch_result(list,task_name )