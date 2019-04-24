import pandas as pd
import click
import math
import time
import utils
from nltk.tokenize import RegexpTokenizer
import gensim


class Context:
    sessionNgrams = []
    sessionWords = []
    log_consulta_query = ""
    log_consulta_query_words = []
    smart_session_id_counter = -1


def different_session(df, i, session_column, row_consulta_query, string1, row_consulta_query_words):
    Context.smart_session_id_counter += 1
    Context.sessionNgrams = []
    Context.sessionWords = []
    if row_consulta_query:
        ngrams = utils.getNgrams(row_consulta_query)
    else:
        ngrams = utils.getNgrams(string1)
    Context.sessionNgrams = utils.mergeNgrams(Context.sessionNgrams, ngrams)
    Context.sessionWords = utils.mergeNgrams(Context.sessionWords, row_consulta_query_words)
    Context.log_consulta_query = row_consulta_query
    Context.log_consulta_query_words = row_consulta_query_words
    df.at[i, session_column] = Context.smart_session_id_counter


def same_session(df, i, session_column, row_consulta_query, string1, row_consulta_query_words):
    if row_consulta_query:
        ngrams = utils.getNgrams(row_consulta_query)
    else:
        ngrams = utils.getNgrams(string1)
    Context.sessionNgrams = utils.mergeNgrams(Context.sessionNgrams, ngrams)
    Context.sessionWords = utils.mergeNgrams(Context.sessionWords, row_consulta_query_words)
    Context.log_consulta_query = row_consulta_query
    Context.log_consulta_query_words = row_consulta_query_words
    df.at[i, session_column] = Context.smart_session_id_counter


def process_data(lim_temp, lim_jac, lim_cosine, lim_wmd, lim_url, model, model_normalized, session_column="SESSION_PRO", user_construct_column="USER_ID"):
    start = time.time()
    names = ["ID", "USER_ID", "QUERY", "URL", "TIMESTAMP", "TIME", "SESSIONID"]
    df = pd.read_csv('aol.csv', sep=';', names=names, index_col="ID")
    click.secho("Preparing...", fg='blue')
    df[session_column] = 0
    number_of_users = 225
    click.secho("Ready!", fg='green')
    tokenizer = RegexpTokenizer(r'\w+')
    with click.progressbar(length=number_of_users, show_pos=True) as progress_bar:
        user_ids = df.USER_ID.unique()
        for user in user_ids:
            progress_bar.update(1)
            df_aux = df[df['USER_ID'] == user]
            df_aux.sort_values('TIME')
            df_aux["DELTA_INSTANTE_TEMPORAL"] = (
                df_aux['TIME'] -
                df_aux['TIME'].shift(1)
            )
            primeira_vez = True
            t_max = 0
            for i in df_aux.index:
                string1 = df_aux.at[i, 'QUERY']
                row_consulta_query = utils.filterString(string1.lower())
                row_consulta_query_words = tokenizer.tokenize(row_consulta_query)
                if primeira_vez:
                    primeira_vez = False
                    if len(df_aux.index) > 1:
                        t_aux = df_aux.loc[df_aux["DELTA_INSTANTE_TEMPORAL"].idxmax()]
                        t_max = int(t_aux["DELTA_INSTANTE_TEMPORAL"])
                    different_session(df, i, session_column, row_consulta_query, string1, row_consulta_query_words)
                else:
                    f_l = -1
                    if len(row_consulta_query) != 0 and len(Context.log_consulta_query) != 0:
                        if row_consulta_query.startswith(Context.log_consulta_query) or row_consulta_query.endswith(Context.log_consulta_query):
                            len_sub_string = utils.size_lcs(Context.log_consulta_query, row_consulta_query)
                            lenn = len(row_consulta_query)
                            if len(Context.log_consulta_query) > lenn:
                                lenn = len(Context.log_consulta_query)
                            k_3_1 = (len_sub_string-1) - (3-1)
                            k_3_2 = lenn - (3-1)
                            k_4_1 = (len_sub_string-1) - (4-1)
                            k_4_2 = lenn - (4-1)
                            f_l = (k_3_1 + k_4_1)/(k_3_2 + k_4_2)
                    delta = df_aux.at[i, 'DELTA_INSTANTE_TEMPORAL']
                    if 2*t_max > 86400:
                        f_t = max(0, 1 - delta/86400)
                    else:
                        f_t = max(0, 1 - delta/(2*t_max))
                    if f_l > math.sqrt((1-(f_t**2))):
                        same_session(df, i, session_column, row_consulta_query, string1, row_consulta_query_words)
                    else:
                        if len(Context.sessionNgrams) > 0:
                            if row_consulta_query:
                                f_l = utils.jaccard_similarity(row_consulta_query, Context.sessionNgrams)
                            else:
                                f_l = utils.jaccard_similarity(string1, Context.sessionNgrams)
                        else:
                            f_l = 1
                        ratioFinal = utils.distanciaLimite(f_t, f_l)
                        if ratioFinal > 1:
                            same_session(df, i, session_column, row_consulta_query, string1, row_consulta_query_words)
                        elif f_t > lim_temp and f_l < lim_jac:
                            try:
                                f_s1 = model.n_similarity(row_consulta_query_words, Context.log_consulta_query_words)
                                if f_s1 > lim_cosine:
                                    same_session(df, i, session_column, row_consulta_query, string1, row_consulta_query_words)
                                else:
                                    f_s2 = model_normalized.wmdistance(row_consulta_query_words, Context.sessionWords)
                                    if f_s2 < lim_wmd:
                                        same_session(df, i, session_column, row_consulta_query, string1, row_consulta_query_words)
                                    elif utils.distanciaLimite(f_s1, f_s2) > 1:
                                        url_row = df_aux.at[i, 'URL']
                                        if not pd.isnull(url_row):
                                            url_log = df_aux.at[i-1, 'URL']
                                            if not pd.isnull(url_log):
                                                url_row_filter = utils.extract_url_domain_name(url_row.lower())
                                                url_log_filter = utils.extract_url_domain_name(url_log.lower())
                                                lennsubstring = utils.size_lcs(url_row_filter, url_log_filter)
                                                lennn = len(url_row_filter)
                                                if len(url_log_filter) > lennn:
                                                    lennn = len(url_log_filter)
                                                simi_sub = lennsubstring/lennn
                                                if simi_sub > lim_url and url_row_filter != '' and url_log_filter != '':
                                                    same_session(df, i, session_column, row_consulta_query, string1, row_consulta_query_words)
                                                else:
                                                    different_session(df, i, session_column, row_consulta_query, string1, row_consulta_query_words)
                                            else:
                                                different_session(df, i, session_column, row_consulta_query, string1, row_consulta_query_words)
                                        else:
                                            different_session(df, i, session_column, row_consulta_query, string1, row_consulta_query_words)
                                    else:
                                        different_session(df, i, session_column, row_consulta_query, string1, row_consulta_query_words)
                            except ZeroDivisionError:
                                different_session(df, i, session_column, row_consulta_query, string1, row_consulta_query_words)
                        else:
                            different_session(df, i, session_column, row_consulta_query, string1, row_consulta_query_words)
    exc_time = int(round((time.time() - start) * 1000))
    print("TIME: ", exc_time)
    df.to_csv('out.csv', sep=';', encoding='utf-8')
    click.secho("EVALUATION!", fg='green')
    N_true_shift = 0
    N_shift_correct = 0
    N_shift = 0
    maxi = 0
    logs_for_ip_iterator = df.iterrows()
    for i, row in logs_for_ip_iterator:
        if row['SESSIONID'] > maxi:
            maxi = row['SESSIONID']
            if row[session_column] != df.loc[i-1, session_column]:
                N_shift_correct += 1
    N = df.loc[df[session_column].idxmax()]
    N_shift = N[session_column]
    N = df.loc[df['SESSIONID'].idxmax()]
    N_true_shift = N['SESSIONID']
    precision = N_shift_correct/N_shift
    recall = N_shift_correct/N_true_shift
    f_measure = 2*(N_shift_correct/(N_shift+N_true_shift))
    ERR = (N_true_shift + N_shift - 2*N_shift_correct)/(N_true_shift + N_shift - N_shift_correct)
    SER = (N_true_shift + N_shift - 2*N_shift_correct)/N_true_shift
    print("PRECISION ", round(precision*100, 2))
    print("RECALL", round(recall*100, 2))
    print("F_MEASURE ", round(f_measure*100, 2))
    print("ERR ", ERR)
    print("SER ", SER)
    print("N_shift_correct ", N_shift_correct)
    print("N_shift ", N_shift)
    print("N_true_shift ", N_true_shift)


def script():
    click.secho("Begin Models...", fg='blue')
    model = gensim.models.keyedvectors.FastTextKeyedVectors.load('wiki_normal.bin')
    click.secho("Process...", fg='blue')
    model_normalized = gensim.models.keyedvectors.FastTextKeyedVectors.load('wiki_with_vectors_normalized.bin')
    process_data(0.7, 0.5, 0.5, 0.1, 0.7, model, model_normalized)

if __name__ == '__main__':
    script()
