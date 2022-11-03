import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pymysql
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import HuberRegressor

warnings.filterwarnings("ignore")

### <DATA IMPORT> ###
order_start_date = (datetime.now() - relativedelta(months=24)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

# get database password
with open('mz_db_password.json', 'rb') as file:
    db_info_dict = json.load(file)

cscart_info = db_info_dict['cscart']
analytics_info = db_info_dict['analytics']

db = pymysql.connect(host=cscart_info['host'], port=cscart_info['port'], user=cscart_info['user'], passwd=cscart_info['passwd'],
                     db=cscart_info['db'])

# order analytics table
anly_data = pd.read_sql('''
SELECT a.order_id, a.user_id, a.product_id, a.product_name_kor, 
    a.option_1_id, a.option_name_1_kor, a.variant_1_id, a.variant_1_name_kor,
    a.option_2_id, a.option_name_2_kor, a.variant_2_id, a.variant_2_name_kor,
    a.barcode, a.category_M, a.category_S, a.currency, a.purchased_at,
    a.product_price, a.marked_down_price, a.product_qty, a.order_qty

FROM cscart_order_analytics a
WHERE a.purchased_at between "''' + order_start_date + '''" and "''' + end_date + '''"
ORDER BY a.purchased_at asc;
''', con=db)

# cart table
cart_data = pd.read_sql('''
SELECT a.user_id, a.type, a.user_type, a.item_id, a.item_type, a.product_id, 
    a.amount, a.price, from_unixtime(a.timestamp+ 14 * 3600, '%Y-%m-%d %H:%i:%s') as add_time
FROM cscart_user_session_products a
WHERE from_unixtime(a.timestamp, '%Y-%m-%d') between "''' + order_start_date + '''" and "''' + end_date + '''"
;
''', con=db)

# user table
user_data = pd.read_sql('''
SELECT a.user_id, a.status, a.user_type, a.company_id, 
    from_unixtime(a.last_login+ 14 * 3600, '%Y-%m-%d %H:%i:%s') as last_login, 
    from_unixtime(a.timestamp+ 14 * 3600, '%Y-%m-%d %H:%i:%s') as register_time, 
    a.firstname, a.lastname, a.birthday, a.age_range, a.gender
FROM cscart_users a
;
''', con=db)

# review table: 기한 제한 없이 처음부터로 수정(21.10.02)
review_data = pd.read_sql('''
SELECT A.post_id, A.thread_id, D.object_id, E.product, 
from_unixtime(A.timestamp+ 14 * 3600, '%Y-%m-%d %H:%i:%s') as post_time,
A.user_id, B.rating_value, C.message, A.status

FROM cscart_discussion_posts A, cscart_discussion_rating B,
cscart_discussion_messages C, 
(SELECT thread_id, object_id, object_type
FROM cscart_discussion
WHERE object_type = 'P') D,
(SELECT product_id, lang_code, product
FROM cscart_product_descriptions
WHERE lang_code = 'ko') E

WHERE 
A.thread_id = B.thread_id AND A.post_id = B.post_id
AND A.post_id = C.post_id AND A.thread_id = C.thread_id
AND A.thread_id = D.thread_id
AND D.object_id = E.product_id

ORDER BY A.timestamp asc;
''', con=db)

# drop education program with products table
products_data = pd.read_sql('''
SELECT a.product_id, a.is_edp, a.status, a.hash_tag
FROM cscart_products a;
''', con=db)
products_data.rename(columns={'product_id': 'object_id'}, inplace=True)

# inventory table
invt_data = pd.read_sql('''
SELECT a.barcode, a.amount, from_unixtime(a.timestamp+ 14 * 3600, '%Y-%m-%d %H:%i:%s') as sys_time
FROM cscart_product_options_inventory_histories a
;
''', con=db)

# brand table
brand_data = pd.read_sql('''
select cscart_product_features_values.product_id, cscart_product_feature_variant_descriptions.variant as brand
from cscart_product_features_values
left join cscart_product_feature_variant_descriptions
on cscart_product_features_values.variant_id = cscart_product_feature_variant_descriptions.variant_id and
cscart_product_features_values.lang_code = cscart_product_feature_variant_descriptions.lang_code
where cscart_product_features_values.feature_id = 19 and cscart_product_features_values.lang_code='ko'
;
''', con=db)
brand_data = brand_data.drop_duplicates(subset='product_id')
brand_data['brand'] = brand_data['brand'].apply(lambda x: x.replace('\t', ''))

# main exposure table
main_data = pd.read_sql('''
select *
from cscart_main_recommend_product_histories
;
''', con=db)

# search table
search_data = pd.read_sql('''
SELECT *, from_unixtime(a.timestamp+ 14 * 3600, '%Y-%m-%d %H:%i:%s') as sys_time
FROM cscart_mz_search_histories a
;
''', con=db)

# cardillo belt 재고 확인 위한 barcode 매칭 테이블
cardillo_bcd = pd.read_sql('''
SELECT *
FROM cscart_product_options_inventory
WHERE product_id=314;
''', con=db)
cardillo_bcd = cardillo_bcd['barcode'].unique()
cardillo_invt = invt_data[invt_data['barcode'].isin(cardillo_bcd)].sort_values(by='sys_time',
                                                                               ascending=False).drop_duplicates(
    subset='barcode')
# 카딜로벨트 필터링 작업 추가
# 1. 옵션 중 하나라도 재고 있으면 메인에 노출시켜야 함
if (cardillo_invt['amount'] > 0).sum() >= 1:
    pass
else:  # 모든 옵션이 재고 없으면 매출테이블에서 아예 카딜로 정보 삭제(추후 연산 안되게)
    anly_data = anly_data[anly_data['product_id'] != 314]

db.close()

## 가격경쟁력 DB
conn = pymysql.connect(host=analytics_info['host'],
                       user=analytics_info['user'],
                       passwd=analytics_info['passwd'],
                       port=analytics_info['port'],
                       db=analytics_info['db'])

## 경쟁사 가격
query = """
SELECT date, competitor, product_id, krw_price, usd_price
FROM monitor_competitorsproduct
WHERE date = (SELECT max(date) FROM monitor_competitorsproduct);
"""
cmpt_price = pd.read_sql(query, conn)
cmpt_price.drop_duplicates(inplace=True)
# cmpt_price = cmpt_price.pivot_table(index='product_id', columns='competitor', values='cmpt_price', aggfunc='min').reset_index()
# cmpt_price.columns = ['product_id']+(cmpt_price.columns[1:]  + '_가격').tolist()

## 수집정보
query = """
SELECT *
FROM monitor_competitors
"""
monitor_competitors = pd.read_sql(query, conn)
monitor_competitors.drop_duplicates(inplace=True)
monitor_competitors.update_date = pd.to_datetime(monitor_competitors.update_date)
# 상품별 가장 최근 수집일 정보
monitor_competitors = monitor_competitors.sort_values(by='update_date', ascending=False).drop_duplicates(
    subset=['competitor', 'product_id'], keep='first')
monitor_competitors = monitor_competitors[['competitor', 'product_id', 'sale_or_not']]
monitor_competitors = monitor_competitors[monitor_competitors['competitor'].isin(cmpt_price['competitor'].unique())]

# 경쟁사 수집정보 종합
merged_cmpt = pd.merge(cmpt_price, monitor_competitors, on=['competitor', 'product_id'], how='outer')
merged_cmpt['수집정보'] = merged_cmpt.apply(
    lambda x: '미판매' if x['sale_or_not'] == 0 else '수집오류' if (x['krw_price'] == -1) | (x['usd_price'] == 0) | (
        pd.isna(x['krw_price'])) else '정상수집', axis=1)
merged_cmpt['cmpt_price'] = merged_cmpt.apply(
    lambda x: '미판매' if x['수집정보'] == '미판매' else '수집오류' if x['수집정보'] == '수집오류' else x['krw_price'], axis=1)

# pivot_table
cmpt_price = merged_cmpt.pivot_table(index='product_id', columns='competitor', values='cmpt_price',
                                     aggfunc='min').reset_index()
cmpt_price.columns = ['product_id'] + (cmpt_price.columns[1:] + '_가격').tolist()
# try:
#     cmpt_price.drop('level_1_가격', axis=1, inplace=True)
# except:
#     pass

## 몬짐 가격
query = """
select product_id, date, krw_price
from monitor_item
join monitor_mzproduct 
on monitor_item.id = monitor_mzproduct.item_id 
where date = (select max(date) from monitor_mzproduct)
;
"""
mz_price = pd.read_sql(query, conn)
mz_price.drop_duplicates(subset='product_id', inplace=True)
mz_price.rename(columns={'krw_price': '몬짐가격'}, inplace=True)
mz_price.drop('date', axis=1, inplace=True)

try:
    price_compare = pd.merge(mz_price, cmpt_price[['product_id', '11st_가격', 'Coupang_가격', 'Iherb_가격', 'Ople_가격']],
                             on='product_id', how='left')
except:  # 가격수집이 되지 않아서 에러 발생할 경우, 전체 컬럼 수집오류로 대체
    price_compare = mz_price.copy()
    price_compare['11st_가격'] = '수집오류'
    price_compare['Coupang_가격'] = '수집오류'
    price_compare['Iherb_가격'] = '수집오류'
    price_compare['Ople_가격'] = '수집오류'
# 없는 컬럼 확인 후 추가해주기. 추후 경쟁사가 바뀔 수 있기 때문에 아래 컬럼명들 모두 바꿔주는게 더 좋을 듯
price_cols = set(['몬짐가격', '11st_가격', 'Coupang_가격', 'Iherb_가격', 'Ople_가격'])
left_cols = list(price_cols - set(price_compare.columns.tolist()))

if len(left_cols) == 0:
    pass
else:
    for left_col in left_cols:
        price_compare[left_col] = np.NaN

price_compare.fillna('미수집', inplace=True)

conn.close()

### <DATA PRERPROCESSING> ###
# 1. change into date type and normalize
anly_data.purchased_at = pd.to_datetime(anly_data.purchased_at).dt.normalize()
cart_data.add_time = pd.to_datetime(cart_data.add_time).dt.normalize()
user_data.register_time = pd.to_datetime(user_data.register_time).dt.normalize()
review_data.post_time = pd.to_datetime(review_data.post_time).dt.normalize()
invt_data.sys_time = pd.to_datetime(invt_data.sys_time).dt.normalize()
search_data.sys_time = pd.to_datetime(search_data.sys_time).dt.normalize()
main_data.date = pd.to_datetime(main_data.date).dt.normalize()

# 2. match column names
review_data.rename(columns={'object_id': 'product_id'})
products_data.rename(columns={'object_id': 'product_id'})

# 3. review text preprocessing
review_data.message = review_data.message.apply(lambda x: str(x))  # 일부 row가 float 형태로 입력
review_data.message = review_data.message.apply(lambda x: x.replace('\r\n', ' '))
review_data.message = review_data.message.apply(lambda x: x.replace('\r', ' '))
review_data.message = review_data.message.apply(lambda x: x.replace('\n', ' '))

# 4. filter category
anly_data = anly_data[~anly_data.category_M.isin(['몬스터짐 코치', '대회/이벤트', '월드오브몬스터짐'])]

# 5. 재고회전잔여일수(invt_turnover_days)
# barcode - product_id matching table
bcd_prod_df = anly_data[['product_id', 'barcode', 'variant_1_name_kor']].drop_duplicates()
# 현재고
now_stock = invt_data[invt_data.sys_time == invt_data.sys_time.max()]
now_stock.rename(columns={'amount': 'stock'}, inplace=True)
# 바코드단위 판매량
avg_qty = anly_data[
    anly_data['purchased_at'].between(anly_data.purchased_at.max() - timedelta(days=90), anly_data.purchased_at.max())]
avg_qty = avg_qty.groupby(['barcode', 'purchased_at'])['product_qty'].sum().groupby('barcode').mean().reset_index()
avg_qty.rename(columns={'product_qty': 'avg_qty'}, inplace=True)
stock_qty = pd.merge(now_stock, avg_qty, on='barcode', how='left')  # 현재고 있는 상품들로만 볼 예정
# barcode and product_id matching
stock_qty = pd.merge(stock_qty, bcd_prod_df, on='barcode', how='inner')  # 역시 재고가 있고, product_id 있는 상품 대상
# product_id별 각 barcode의 판매량 비율
prod_barcode_rate = (
        stock_qty.groupby(['product_id', 'barcode'])['avg_qty'].first() / stock_qty.groupby(['product_id'])[
    'avg_qty'].sum()).reset_index()
prod_barcode_rate.rename(columns={'avg_qty': 'prod_bcd_rate'}, inplace=True)
stock_qty = pd.merge(stock_qty, prod_barcode_rate)
# barcode별 inventory turnover days
stock_qty['avg_qty'].fillna(1, inplace=True)
stock_qty['ivt_turnover_days'] = stock_qty.stock / stock_qty.avg_qty
stock_qty = stock_qty[
    ['sys_time', 'product_id', 'barcode', 'variant_1_name_kor', 'stock', 'avg_qty', 'ivt_turnover_days',
     'prod_bcd_rate']]
# ==================================
# 상품 하나에 재고와 판매가 각각 다른 product_id로 관리되고 있어서 임의로 변경
stock_qty['product_id'] = stock_qty['product_id'].astype('str')
stock_qty['product_id'] = stock_qty['product_id'].apply(lambda x: x.replace('4241', '2980'))
stock_qty['product_id'] = stock_qty['product_id'].astype('int')
# ==================================
bcd_cnt = stock_qty.groupby('product_id')['barcode'].count().reset_index()
only_prod = bcd_cnt[bcd_cnt.barcode == 1].product_id.tolist()
multi_prod = bcd_cnt[bcd_cnt.barcode != 1].product_id.tolist()
# 1) 단일상품인경우
only_bcd = stock_qty[stock_qty.product_id.isin(only_prod)]
only_bcd = only_bcd[['product_id', 'barcode', 'variant_1_name_kor', 'prod_bcd_rate', 'stock', 'ivt_turnover_days']]
only_bcd.rename(columns={
    'barcode': '1st판매옵션바코드',
    'variant_1_name_kor': '1st판매옵션명',
    'stock': '1st판매옵션현재고',
    'ivt_turnover_days': '1st판매옵션재고전환일수',
    'prod_bcd_rate': '1st판매옵션비율'
}, inplace=True)
# 2) 상품유형에 여러 바코드가 있는 상품인 경우
# 한 상품 내에서 가장 많이 팔리는 바코드
first_bcd = stock_qty[stock_qty.product_id.isin(multi_prod)].sort_values(by='prod_bcd_rate',
                                                                         ascending=False).drop_duplicates(
    subset='product_id', keep='first')
first_bcd = first_bcd[['product_id', 'barcode', 'variant_1_name_kor', 'prod_bcd_rate', 'stock', 'ivt_turnover_days']]
first_bcd.rename(columns={
    'barcode': '1st판매옵션바코드',
    'variant_1_name_kor': '1st판매옵션명',
    'stock': '1st판매옵션현재고',
    'ivt_turnover_days': '1st판매옵션재고전환일수',
    'prod_bcd_rate': '1st판매옵션비율'
}, inplace=True)
first_bcd_idx = first_bcd.index
# 한 상품 내에서 두번째로 많이 팔리는 바코드
second_stock_qty = stock_qty.drop(first_bcd_idx)
second_bcd = second_stock_qty[second_stock_qty.product_id.isin(multi_prod)].sort_values(by='prod_bcd_rate',
                                                                                        ascending=False).drop_duplicates(
    subset='product_id', keep='first')
second_bcd = second_bcd[['product_id', 'barcode', 'variant_1_name_kor', 'prod_bcd_rate', 'stock', 'ivt_turnover_days']]
second_bcd.rename(columns={
    'barcode': '2nd판매옵션바코드',
    'variant_1_name_kor': '2nd판매옵션명',
    'stock': '2nd판매옵션현재고',
    'ivt_turnover_days': '2nd판매옵션재고전환일수',
    'prod_bcd_rate': '2nd판매옵션비율'
}, inplace=True)
multi_bcd = pd.merge(first_bcd, second_bcd, on='product_id')
final_bcd_stock = pd.concat([only_bcd, multi_bcd], sort=False)
# 자리수 줄이기
final_bcd_stock['1st판매옵션비율'] = np.round(final_bcd_stock['1st판매옵션비율'], 2)
final_bcd_stock['1st판매옵션재고전환일수'] = np.round(final_bcd_stock['1st판매옵션재고전환일수'], 2)
final_bcd_stock['2nd판매옵션비율'] = np.round(final_bcd_stock['2nd판매옵션비율'], 2)
final_bcd_stock['2nd판매옵션재고전환일수'] = np.round(final_bcd_stock['2nd판매옵션재고전환일수'], 2)
final_bcd_stock['2nd판매옵션재고전환일수'].fillna(0, inplace=True)

# 6. 메인페이지 노출정보 및 노출후상승률
# 가장 최근 노출된 상품들을 기준으로, 해당 노출이 연속적으로 이어진 처음 시작 날짜를 반목문을 통해 찾아가는 과정
latest_exposure = main_data[main_data['date'] == main_data['date'].max()]
latest_exposure.sort_values(by='popularity', ascending=False, inplace=True)
latest_exposure = latest_exposure.drop_duplicates(subset='product_id')

latest_exposed_items = latest_exposure.product_id.tolist()
start_exposure = latest_exposure[['product_id', 'date']].set_index('product_id').to_dict()['date']
latest_date = main_data['date'].max()
day_diff = 1
while len(latest_exposed_items) != 0:
    check_date = latest_date - timedelta(days=day_diff)
    temp_main = main_data[main_data['date'] == check_date]
    temp_main = temp_main[temp_main['product_id'].isin(latest_exposed_items)]
    changed_start_prods = temp_main['product_id'].unique()
    for prod in changed_start_prods:
        start_exposure[prod] = check_date
    latest_exposed_items = changed_start_prods
    day_diff += 1

start_exposure = pd.DataFrame.from_dict(start_exposure, orient='index').reset_index()
start_exposure.columns = ['product_id', 'exposure_start_date']
start_exposure = pd.merge(latest_exposure[['product_id', 'main_rec_category']], start_exposure, on='product_id',
                          how='left')

# 확인한 가장최근노출시작일자를 바탕으로 노출전후를 비교. 실험군은 연속적인 최근노출기간 일평균판매량, 대조군은 노출시작 직전 일주일 일평균판매량.
latest_exposed_items = latest_exposure.product_id.tolist()  # 위에서 다 없앴기 때문에 다시 생성
whole_exposed_anly = anly_data[anly_data['product_id'].isin(latest_exposed_items)]
whole_exposed_anly = whole_exposed_anly.groupby(['product_id', 'purchased_at'])['product_qty'].sum().reset_index()

tot_effect = []
for prod in latest_exposed_items:
    prod_start_date = start_exposure[start_exposure['product_id'] == prod]['exposure_start_date'].values[0]
    prod_anly = whole_exposed_anly[whole_exposed_anly['product_id'] == prod]
    exposed_anly = prod_anly[prod_anly['purchased_at'].between(prod_start_date, latest_date, inclusive='both')]
    compare_anly = prod_anly[
        prod_anly['purchased_at'].between(pd.to_datetime(prod_start_date) - timedelta(days=7), prod_start_date,
                                          inclusive='left')]
    # 실험군, 대조군 각각 기간에 판매량이 0일 수 있기 때문에 0처리
    if len(exposed_anly) == 0:
        exposed_qty = 0
    else:
        whole_sale_days = (latest_date - prod_start_date).days + 1
        exposed_qty = exposed_anly['product_qty'].sum() / whole_sale_days
    if len(compare_anly) == 0:
        compare_qty = 0 + 1  # 분모이기 때문에 1로 수정
    else:
        whole_sale_days = (pd.to_datetime(prod_start_date) - (
                pd.to_datetime(prod_start_date) - timedelta(days=7))).days + 1
        compare_qty = compare_anly['product_qty'].sum() / whole_sale_days
    exposure_effect = exposed_qty / compare_qty
    tot_effect.append(pd.DataFrame([prod, exposure_effect]).T)
tot_effect = pd.concat(tot_effect)
tot_effect.columns = ['product_id', 'exposed_effect']
tot_effect['exposed_effect'] = np.round(tot_effect['exposed_effect'], 2)

exposed_data = pd.merge(start_exposure, tot_effect, on='product_id')
exposed_data.columns = ['product_id', '최근노출영역', '연속노출시작일자', '노출후상승률']

# 필터링할 브랜드 명
filtered_brand = ['Jan Tana']


### FUNCTIONS ###
# 1. 급상승함수
def convert_month(x, first_month):
    return (x.year - first_month.year) * 12 + (x.month - first_month.month)


# 장기상승률(LTGR)
def process(x):
    if len(x) < 3:  # ABNORMAL_TERMINATION_IN_LNSRCH 에러 때문에 3으로 변경
        return 0
    else:
        a = np.array(x['month_num']).reshape(-1, 1)
        #     b = np.array(x['sum'])
        smax = x['product_qty'].max()
        b = np.array([i / smax for i in list(x['product_qty'])])
        huber = HuberRegressor().fit(a, b)
        return huber.coef_[0]


def conditional_filtering(df, condition_prods=[314], min_turnover=21):
    '''
    카딜로 이슈처럼 사전에 필터링 하지 못하고, 특정 상품들을 위해 후처리 필터링 하는 임시 함수(22.08.28)
    '''
    # 1. 재고전환일수 필터링
    cond_df = df[df['product_id'].isin(condition_prods)]  # 제고 고려 제외하는 상품들
    non_stock_df = df[df['1st판매옵션재고전환일수'].isnull()]  # 재고정보 없는 한국제품들
    df = df[(df['1st판매옵션재고전환일수'] + df['2nd판매옵션재고전환일수']) >= min_turnover]
    df = pd.concat([df, non_stock_df, cond_df], ignore_index=True)
    df.sort_values(by='판매량상승률', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2. 메인페이지 노출 후 상승률이 1(100%) 이상인 상품은 이번주도 노출 대상에 포함시킨다.
    cond_df = df[df['product_id'].isin(condition_prods)]
    unexposed_df = df[df['최근노출영역'].isnull()]
    exposed_df = df[df['최근노출영역'].notnull()]
    exposed_df = exposed_df[exposed_df['노출후상승률'] >= 1]
    df = pd.concat([unexposed_df, exposed_df, cond_df])
    return df


def filtering_rap_increase(final_df):
    '''
    (삭제)조건0. 1st, 2nd 옵션의 판매비중 합이 0.5 이하인 골고루 팔리는 상품은 재고필터링에서 제외한다.(22.02.14추가)
    (작업위치변경)조건1. 1st, 2nd 옵션의 재고 전환일수가 적어도 둘 중 하나는 DIO_days 이상이어야 한다.
    (작업위치변경)조건2. 메인페이지 노출 후 상승률이 1(100%) 이상인 상품은 이번주도 노출 대상에 포함시킨다.
    조건3. 해당 상품들을 판매량 상승률 기준으로 정렬한다.
    '''

    col_names = final_df.columns
    df = final_df.drop(0, axis=0)

    #     # 조건2. 메인페이지 노출 후 상승률이 1(100%) 이상인 상품은 이번주도 노출 대상에 포함시킨다.
    #     unexposed_df = df[df['최근노출영역'].isnull()]
    #     exposed_df = df[df['최근노출영역'].notnull()]
    #     exposed_df = exposed_df[exposed_df['노출후상승률'] >= 1]
    #     df = pd.concat([unexposed_df, exposed_df])
    df = conditional_filtering(df)
    # 조건3. 해당 상품들을 판매량 상승률 기준으로 정렬한다.
    df = df.sort_values(by='판매량상승률', ascending=False).reset_index(drop=True)

    # benchmark
    benchmark = pd.DataFrame(df.mean()).T
    benchmark.index = ['benchmark']
    benchmark['product_id'] = np.nan
    benchmark['1st판매옵션바코드'] = np.nan
    benchmark['2nd판매옵션바코드'] = np.nan
    benchmark = np.round(benchmark, 2)
    df = pd.concat([benchmark, df], sort=False)
    df = df[col_names]  # 컬럼 순서 변경

    return df


def rap_increase_prod(anly_data, final_bcd_stock, invt_data, bcd_prod_df, brand_data, main_data, exposed_data,
                      price_compare,
                      today='latest', latest_Ndays=7, compare_Ndays=28, atleast_avgqty=0,
                      atleast_Nqty=6, head_num=None):
    '''
    anly_data: 판매 데이터
    final_bcd_stock: 위의 0.공통처리한 상품별 재고정보 담고있는 테이블
    invt_data: 재고 테이블
    bcd_prod_df: barcode와 product_id 매칭 테이블
    brand_data: 브랜드 정보 매칭 테이블
    main_data: MZ 메인페이지 노출정보 담은 테이블
    today: 분석에 기준이 될 가장 최근 시점의 날짜 '2021-01-01' 형태로 넣으면 됨. default는 돌리는 시점의 가장 최근일자
    latest_Ndays: 최근 며칠을 기준으로 급상승 판단할 것인지
    compare_Ndays: latest_Ndays와 비교할 과거 며칠간의 기간을 제한하는지
    atleast_avgqty: 일평균판매량 몇개 이상인지 threshold
    atleast_Nqty: latest_Ndays 기간동안 적어도 N개 이상 팔린 상품들에 대해서만 필터링
    head_num: 최대 몇개까지 상품 볼건지
    min_turnover: 최소한 보장되어야하는 재고전환일수 기준
    '''

    # 장기상승률 추가
    anly_data['year_month'] = anly_data.purchased_at.apply(lambda x: datetime(x.year, x.month, 1))
    monthly_qty = anly_data.groupby(['product_id', 'year_month']).product_qty.sum().reset_index()
    first_month = monthly_qty.year_month.min()
    monthly_qty['month_num'] = monthly_qty['year_month'].apply(lambda x: convert_month(x, first_month))
    p_slopes = monthly_qty.groupby(['product_id']).apply(process).reset_index()
    p_slopes.rename(columns={0: 'LTGR_coef'}, inplace=True)
    p_slopes['LTGR_coef'] = np.round(p_slopes['LTGR_coef'], 2)

    # → 여기서부터 원래 급상승
    if today == 'latest':
        today = anly_data.purchased_at.max()
    else:
        today = datetime.strptime(today, '%Y-%m-%d')

    latest_df = anly_data[anly_data.purchased_at.between(today - timedelta(days=latest_Ndays), today)]
    compare_df = anly_data[
        anly_data.purchased_at.between(today - timedelta(days=latest_Ndays) - timedelta(days=compare_Ndays),
                                       today - timedelta(days=latest_Ndays))]

    # 0판매 복구시켜줄 재고 확인 테이블
    stock_left = invt_data[invt_data.amount != 0]  # 재고가 0이 아닌 테이블
    stock_left = pd.merge(stock_left, bcd_prod_df[['barcode', 'product_id']], on='barcode', how='left')
    stock_left = stock_left.groupby(['product_id', 'sys_time'])['amount'].sum().reset_index()
    stock_left.rename(columns={'sys_time': 'purchased_at'}, inplace=True)
    stock_left.drop('amount', axis=1, inplace=True)

    # 그냥 groupby mean을 하면 한 거래단위의 평균값이 나오기 때문에 거의 한 두개. 따라서 일단위 합계를 구한 뒤 평균을 내야 함
    latest_df = latest_df.groupby(['product_id', 'purchased_at'])['product_qty'].sum().reset_index()
    # latest_Ndays 기간동안 적어도 N개 이상 팔린 상품들에 대해서만 필터링
    atleast_df = latest_df.groupby('product_id')['product_qty'].sum()
    atleast_list = atleast_df[atleast_df > atleast_Nqty].index.tolist()
    latest_df = latest_df[latest_df.product_id.isin(atleast_list)]
    latest_df = pd.merge(latest_df, stock_left, on=['product_id', 'purchased_at'], how='outer')
    latest_df.product_qty.fillna(0, inplace=True)
    latest_df = latest_df[latest_df.purchased_at.between(today - timedelta(days=latest_Ndays), today)]
    latest_df = latest_df.groupby('product_id')['product_qty'].mean().reset_index()
    latest_df.rename(columns={'product_qty': 'latest_qty'}, inplace=True)

    compare_df = compare_df.groupby(['product_id', 'purchased_at'])['product_qty'].sum().reset_index()
    compare_df = pd.merge(compare_df, stock_left, on=['product_id', 'purchased_at'], how='outer')
    compare_df.product_qty.fillna(0, inplace=True)
    compare_df = compare_df[
        compare_df.purchased_at.between(today - timedelta(days=latest_Ndays) - timedelta(days=compare_Ndays),
                                        today - timedelta(days=latest_Ndays))]
    compare_df = compare_df.groupby('product_id')['product_qty'].mean().reset_index()
    compare_df.rename(columns={'product_qty': 'compare_qty'}, inplace=True)
    compare_df.compare_qty = compare_df.compare_qty.apply(lambda x: 1 if x < 1 else x)  # 분모가 1보다 작으면 1로 보정

    merged_df = pd.merge(latest_df, compare_df, how='left')  # 대조군에 없어서 분모가 0이되면 무한대가 되니 제외
    merged_df.compare_qty.fillna(1, inplace=True)  # 과거에 판매가 없었다면 0인데, 분모가 될 수 없으니 1로 보정
    merged_df['increase_rate'] = merged_df['latest_qty'] / merged_df['compare_qty']
    merged_df.sort_values(by='increase_rate', ascending=False, inplace=True)
    merged_df = merged_df[merged_df.latest_qty >= atleast_avgqty]

    merged_df = pd.merge(merged_df, anly_data[['product_id', 'product_name_kor']].drop_duplicates(), how='left')
    merged_df = pd.merge(merged_df, brand_data, on='product_id', how='left')
    merged_df = pd.merge(merged_df, p_slopes, on='product_id', how='left')
    merged_df = merged_df[~merged_df['brand'].isin(filtered_brand)]
    #     merged_df = merged_df[['product_id', 'product_name_kor', 'brand', 'latest_qty', 'compare_qty', 'increase_rate']]

    # 자릿수 줄이기
    merged_df['latest_qty'] = np.round(merged_df['latest_qty'], 2)
    merged_df['compare_qty'] = np.round(merged_df['compare_qty'], 2)
    merged_df['increase_rate'] = np.round(merged_df['increase_rate'], 2)

    merged_df.rename(columns={
        'product_name_kor': '상품명',
        'latest_qty': '최근' + str(latest_Ndays) + '일 일평균판매량',
        'compare_qty': '과거' + str(compare_Ndays) + '일 일평균판매량',
        'increase_rate': '판매량상승률',
        'LTGR_coef': '장기상승률'
    }, inplace=True)

    final_df = pd.merge(merged_df, final_bcd_stock, how='outer')
    final_df_cols = final_df.columns.tolist()

    # 컬럼순서 변경
    final_df = final_df[final_df_cols]
    final_df = final_df[final_df['최근7일 일평균판매량'].notnull()]

    # 경쟁사 가격
    final_df = pd.merge(final_df, price_compare, on='product_id', how='left')

    # benchmark
    benchmark = pd.DataFrame(final_df.mean()).T
    benchmark.index = ['benchmark']
    benchmark['product_id'] = np.nan
    benchmark['1st판매옵션바코드'] = np.nan
    benchmark['2nd판매옵션바코드'] = np.nan
    benchmark = np.round(benchmark, 2)
    final_df = pd.concat([benchmark, final_df], sort=False)

    final_df = final_df[[
        'brand', '상품명', '판매량상승률', '최근' + str(latest_Ndays) + '일 일평균판매량',
                                  '과거' + str(compare_Ndays) + '일 일평균판매량',
        '몬짐가격', '11st_가격', 'Coupang_가격', 'Iherb_가격', 'Ople_가격',
        '장기상승률',
        '1st판매옵션명', '1st판매옵션현재고', '1st판매옵션재고전환일수', '1st판매옵션비율',
        '2nd판매옵션명', '2nd판매옵션현재고', '2nd판매옵션재고전환일수', '2nd판매옵션비율',
        'product_id', '1st판매옵션바코드', '2nd판매옵션바코드',

    ]]

    final_df = pd.merge(final_df, exposed_data, on='product_id', how='left')

    final_df = filtering_rap_increase(final_df)
    final_df.drop_duplicates(subset='product_id', inplace=True)
    if head_num == None:
        final_df = final_df
    else:
        final_df = final_df.head(head_num + 1)

    final_df.index = ['benchmark'] + list(range(1, len(final_df)))

    return final_df


# 2. 재구매 함수
def filtering_repurchase(final_df, DIO_days):
    '''
    조건0. 1st, 2nd 옵션의 판매비중 합이 0.5 이하인 골고루 팔리는 상품은 재고필터링에서 제외한다.(22.02.14추가)
    조건1. 1st, 2nd 옵션의 재고 전환일수가 적어도 둘 중 하나는 DIO_days 이상이어야 한다.
    조건2. 메인페이지 노출 후 상승률이 1(100%) 이상인 상품은 이번주도 노출 대상에 포함시킨다.
    조건3. 해당 상품들을 재구매지수 기준으로 정렬한다.
    '''

    col_names = final_df.columns
    df = final_df.drop(0, axis=0)

    # 조건0, 조건1
    cardillo_stock = df[df['product_id'] == 314]  # 카딜로는 제고 고려 제외
    non_stock_df = df[df['1st판매옵션재고전환일수'].isnull()]  # 재고정보 없는 한국제품들
    #     even_sell_option_df = df[
    #         (df['1st판매옵션비율'] + df['2nd판매옵션비율']) <= 0.5]  # 옵션이 골고루 판매되는 상품은 1, 2번째 옵션 재고가 적더라도 그 이후 재고가 많으면 보여질 수 있음
    df = df[(df['1st판매옵션재고전환일수'] + df['2nd판매옵션재고전환일수']) >= DIO_days]
    #     df = df[(df['1st판매옵션재고전환일수']>=DIO_days)|(df['2nd판매옵션재고전환일수']>=DIO_days)]
    #     df = pd.concat([df, non_stock_df, even_sell_option_df, cardillo_stock], ignore_index=True)
    df = pd.concat([df, non_stock_df, cardillo_stock], ignore_index=True)
    df.reset_index(drop=True, inplace=True)

    #     # 조건1. 1st, 2nd 옵션의 재고 전환일수가 적어도 둘 중 하나는 DIO_days 이상이어야 한다.
    #     df = df[(df['1st판매옵션재고전환일수']>=DIO_days)|(df['2nd판매옵션재고전환일수']>=DIO_days)]

    # 조건2. 메인페이지 노출 후 상승률이 1(100%) 이상인 상품은 이번주도 노출 대상에 포함시킨다.
    unexposed_df = df[df['최근노출영역'].isnull()]
    exposed_df = df[df['최근노출영역'].notnull()]
    exposed_df = exposed_df[exposed_df['노출후상승률'] >= 1]
    df = pd.concat([unexposed_df, exposed_df])

    # 조건3. 해당 상품들을 최종점수률 기준으로 정렬한다.
    df = df.sort_values(by='최종점수', ascending=False).reset_index(drop=True)

    # benchmark
    benchmark = pd.DataFrame(df.mean()).T
    benchmark.index = ['benchmark']
    benchmark['product_id'] = np.nan
    benchmark['1st판매옵션바코드'] = np.nan
    benchmark['2nd판매옵션바코드'] = np.nan
    benchmark = np.round(benchmark, 2)
    df = pd.concat([benchmark, df], sort=False)
    df = df[col_names]  # 컬럼 순서 변경

    return df


# 2. 재구매율 높은 상품
def repurchase_prod(anly_data, final_bcd_stock, brand_data, main_data, exposed_data, price_compare,
                    today='latest', target_period=90, min_repurchase=30,
                    min_repurchase_user=10, latest_Ndays=7, head_num=None, min_turnover=21):
    '''
    anly_data: 판매 데이터
    final_bcd_stock: 위의 0.공통처리한 상품별 재고정보 담고있는 테이블
    main_data: MZ 메인페이지 노출정보 담은 테이블
    today: 분석에 기준이 될 가장 최근 시점의 날짜 '2021-01-01' 형태로 넣으면 됨. default는 돌리는 시점의 가장 최근일자
    target_period: 분석 대상 기간. today를 기준으로 며칠간의 기간 중 재구매를 볼 것인지
    min_repurchase: 최소 재구매건수
    min_repurchase_user: 최소 재구매유저수
    latest_Ndays: 최근 며칠 판매량을 반영할 것인지
    head_num: 최대 몇개까지 상품 볼건지
    min_turnover: 최소한 보장되어야하는 재고전환일수 기준
    '''

    if today == 'latest':
        today = anly_data.purchased_at.max()
    else:
        today = datetime.strptime(today, '%Y-%m-%d')

    anly_data = anly_data[anly_data.purchased_at.between(today - timedelta(days=target_period), today)]

    # 1) 구매건수기반의 재구매 상품(재구매건수/전체구매건수)
    tot_pchs = anly_data.groupby('product_id')['user_id'].count().reset_index()
    tot_pchs.rename(columns={'user_id': 'total_purchase'}, inplace=True)
    fst_pchs = anly_data.groupby('product_id')['user_id'].nunique().reset_index()
    fst_pchs.rename(columns={'user_id': 'first_purchase'}, inplace=True)
    repurchase_rate = pd.merge(tot_pchs, fst_pchs, on='product_id', how='inner')
    repurchase_rate['repurchase'] = repurchase_rate.total_purchase - repurchase_rate.first_purchase
    repurchase_rate['repurchase_rate'] = repurchase_rate.repurchase / repurchase_rate.total_purchase
    repurchase_rate.sort_values(by='repurchase_rate', ascending=False, inplace=True)
    repurchase_rate = pd.merge(repurchase_rate, anly_data[['product_id', 'product_name_kor']].drop_duplicates(),
                               how='left')
    repurchase_rate = repurchase_rate[
        ['product_id', 'product_name_kor', 'repurchase', 'total_purchase', 'repurchase_rate']]

    # 자릿수 줄이기
    repurchase_rate['repurchase_rate'] = np.round(repurchase_rate['repurchase_rate'], 2)
    repurchase_rate.rename(columns={
        'product_name_kor': '상품명',
        'total_purchase': '전체구매건수',
        'repurchase': '재구매건수',
        'repurchase_rate': '재구매건수비율'
    }, inplace=True)

    # 2) 구매유저수기반의 재구매 상품(재구매한유저수/전체구매유저수)
    anly_data = anly_data.sort_values(by=['user_id', 'purchased_at'])
    first_order = anly_data.drop_duplicates(subset=['user_id', 'product_id'], keep='first')  # 유저별, 상품별로 첫 구매 기록
    reanly_data = anly_data.drop(first_order.index, axis=0)  # 첫구매 제외한 재구매 데이터들만
    # 재구매건수 중 unique한 유저의 비율(즉, 1에 가까울수록 general한 구매, 0에 가까울수록 maniac한 구매)
    retot_pchs = reanly_data.groupby('product_id')['user_id'].count().reset_index()
    retot_pchs.rename(columns={'user_id': 'total_repurchase'}, inplace=True)
    unique_users = reanly_data.groupby('product_id')['user_id'].nunique().reset_index()
    unique_users.rename(columns={'user_id': 'unique_users'}, inplace=True)
    general_rate = pd.merge(retot_pchs, unique_users, on='product_id', how='inner')
    general_rate['general_repurchase_rate'] = general_rate.unique_users / general_rate.total_repurchase
    general_rate = general_rate[general_rate.total_repurchase >= min_repurchase]
    general_rate = general_rate[general_rate.unique_users >= min_repurchase_user]
    general_rate.sort_values(by='general_repurchase_rate', ascending=False, inplace=True)
    general_rate = pd.merge(general_rate,
                            anly_data[['product_id', 'product_name_kor', 'category_M', 'category_S']].drop_duplicates(),
                            how='left')
    general_rate = pd.merge(general_rate, brand_data, on='product_id', how='left')
    general_rate = general_rate[~general_rate['brand'].isin(filtered_brand)]
    general_rate = general_rate[
        ['product_id', 'product_name_kor', 'brand', 'category_M', 'category_S', 'unique_users', 'total_repurchase',
         'general_repurchase_rate']]

    # 자릿수 줄이기
    general_rate['general_repurchase_rate'] = np.round(general_rate['general_repurchase_rate'], 2)
    general_rate.rename(columns={
        'category_M': '메인카테고리',
        'category_S': '서브카테고리',
        'product_name_kor': '상품명',
        'unique_users': '재구매유저수',
        'total_repurchase': '재구매건수',
        'general_repurchase_rate': '재구매건수중유저비율'
    }, inplace=True)

    # 두 재구매율 함께 고려
    repurchase_df = pd.merge(repurchase_rate, general_rate)
    repurchase_df['재구매지수'] = repurchase_df['재구매건수비율'] * repurchase_df['재구매건수중유저비율']

    final_df = pd.merge(repurchase_df, final_bcd_stock, how='outer')
    #     if in_stock==True:
    #         final_df = final_df[final_df['1st판매옵션현재고']>0]
    #     else:
    #         pass
    final_df_cols = final_df.columns.tolist()

    # 재고전환일수 필터링
    cardillo_stock = final_df[final_df['product_id'] == 314]  # 카딜로는 제고 고려 제외
    non_stock_df = final_df[final_df['1st판매옵션재고전환일수'].isnull()]  # 재고정보 없는 한국제품들
    #     even_sell_option_df = final_df[(final_df['1st판매옵션비율'] + final_df[
    #         '2nd판매옵션비율']) <= 0.5]  # 옵션이 골고루 판매되는 상품은 1, 2번째 옵션 재고가 적더라도 그 이후 재고가 많으면 보여질 수 있음
    final_df = final_df[(final_df['1st판매옵션재고전환일수'] + final_df['2nd판매옵션재고전환일수']) >= min_turnover]
    #     final_df = final_df[(final_df['1st판매옵션재고전환일수']>=min_turnover)|(final_df['2nd판매옵션재고전환일수']>=min_turnover)]
    #     final_df = pd.concat([final_df, non_stock_df, even_sell_option_df, cardillo_stock], ignore_index=True)
    final_df = pd.concat([final_df, non_stock_df, cardillo_stock], ignore_index=True)
    final_df.sort_values(by='재구매지수', ascending=False, inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    # 원래 컬럼순서대로 변경
    final_df = final_df[final_df_cols]
    final_df = final_df[final_df['재구매지수'].notnull()]

    # 경쟁사 가격
    final_df = pd.merge(final_df, price_compare, on='product_id', how='left')

    # 가중치를 위한 최근 N일 평균 판매량 변수 추가(22.03.26)
    latest_df = anly_data[anly_data.purchased_at.between(today - timedelta(days=latest_Ndays), today)]
    latest_df = latest_df.groupby(['product_id', 'purchased_at'])['product_qty'].sum().reset_index()
    latest_df = latest_df.groupby('product_id')['product_qty'].mean().reset_index()
    latest_df['product_qty'] = np.round(latest_df['product_qty'], 2)
    latest_df.rename(columns={'product_qty': '최근' + str(latest_Ndays) + '일 일평균판매량'}, inplace=True)

    final_df = pd.merge(final_df, latest_df, on='product_id', how='left')
    final_df = final_df[~final_df['최근' + str(latest_Ndays) + '일 일평균판매량'].isna()]

    # 컬럼정렬 1
    final_df = final_df[[
        'brand', '상품명',
        '재구매지수',
        '몬짐가격', '11st_가격', 'Coupang_가격', 'Iherb_가격', 'Ople_가격',
        '재구매건수비율', '재구매건수중유저비율', '재구매건수', '전체구매건수', '재구매유저수',
        '최근' + str(latest_Ndays) + '일 일평균판매량',
        '메인카테고리', '서브카테고리',
        '1st판매옵션명', '1st판매옵션현재고', '1st판매옵션재고전환일수', '1st판매옵션비율',
        '2nd판매옵션명', '2nd판매옵션현재고', '2nd판매옵션재고전환일수', '2nd판매옵션비율',
        'product_id', '1st판매옵션바코드', '2nd판매옵션바코드',
    ]]

    # 점수화 과정
    plus_cols = ['재구매건수비율', '재구매건수중유저비율', '최근' + str(latest_Ndays) + '일 일평균판매량']
    grade_cols = []
    for col in plus_cols:
        temp_qtl = []
        temp_series = final_df[col].copy()
        for num in range(10, 1, -1):
            term = (temp_series.max() - temp_series.min()) / num
            term = temp_series.max() - term
            temp_series = temp_series[temp_series < term]
            temp_qtl.append(term)
        temp_qtl.reverse()
        col_name = col + '_점수'
        final_df[col_name] = final_df[col].apply(lambda x: get_grade(x, temp_qtl))
        grade_cols.append(col_name)
        # 판매량 가중치 점수 비중 축소
    final_df['최근' + str(latest_Ndays) + '일 일평균판매량' + '_점수'] = final_df[
                                                                  '최근' + str(latest_Ndays) + '일 일평균판매량' + '_점수'] / 2
    # 점수 합산
    final_df['최종점수'] = final_df[grade_cols].sum(axis=1)

    # benchmark
    benchmark = pd.DataFrame(final_df.mean()).T
    benchmark.index = ['benchmark']
    benchmark['product_id'] = np.nan
    benchmark['1st판매옵션바코드'] = np.nan
    benchmark['2nd판매옵션바코드'] = np.nan
    benchmark = np.round(benchmark, 2)
    final_df = pd.concat([benchmark, final_df], sort=False)

    # 컬럼정렬 2
    org_cols = final_df.columns.tolist()
    score_cols = ['최종점수'] + grade_cols
    prod_cols = ['brand', '상품명']
    rest_cols = [col for col in org_cols if col not in score_cols + prod_cols]
    tot_cols = prod_cols + score_cols + rest_cols
    final_df = final_df[tot_cols]

    final_df = pd.merge(final_df, exposed_data, on='product_id', how='left')
    DIO_days = min_turnover
    final_df = filtering_repurchase(final_df, DIO_days)
    final_df.drop_duplicates(subset='product_id', inplace=True)
    if head_num == None:
        final_df = final_df
    else:
        final_df = final_df.head(head_num + 1)
    final_df.index = ['benchmark'] + list(range(1, len(final_df)))
    #     # file save
    #     today_str = datetime.today().strftime('%Y%m%d')
    #     file_name = '재구매_' + today_str + '.xlsx'
    #     final_df.to_excel(file_name)

    return final_df


# 3. 랭킹 함수
def get_grade(value, qtl_list, reverse=False):
    if value <= qtl_list[0]:
        result = 1
    elif (value > qtl_list[0]) & (value <= qtl_list[1]):
        result = 2
    elif (value > qtl_list[1]) & (value <= qtl_list[2]):
        result = 3
    elif (value > qtl_list[2]) & (value <= qtl_list[3]):
        result = 4
    elif (value > qtl_list[3]) & (value <= qtl_list[4]):
        result = 5
    elif (value > qtl_list[4]) & (value <= qtl_list[5]):
        result = 6
    elif (value > qtl_list[5]) & (value <= qtl_list[6]):
        result = 7
    elif (value > qtl_list[6]) & (value <= qtl_list[7]):
        result = 8
    elif (value > qtl_list[7]) & (value <= qtl_list[8]):
        result = 9
    else:
        result = 10
    if reverse == False:
        return result
    else:
        return 11 - result


def get_final_score(df, plus_cols, minus_dict=None, plus_dict=None, filename_prefix=None, head_num=50):
    '''
    df: 점수 연산할 raw data의 데이터프레임
    plus_cols: 가점 계산할 대상이 되는 컬럼명
    plus_dict: 가중치 줄 컬럼의 딕셔너리
    minus_dict: 감점 줄 컬럼의 딕셔너리
    filename_prefix: 산출물 파일 저장할 때 파일명 앞에 붙일 prefix
    '''
    grade_cols = []
    temp_final = df.copy()

    for col in plus_cols:
        temp_qtl = []
        temp_series = temp_final[col].copy()
        for num in range(10, 1, -1):
            term = (temp_series.max() - temp_series.min()) / num
            term = temp_series.max() - term
            temp_series = temp_series[temp_series < term]
            temp_qtl.append(term)
        temp_qtl.reverse()
        col_name = col + '_점수'
        if col == 'OOS_RATE':
            temp_final[col_name] = temp_final[col].apply(lambda x: get_grade(x, temp_qtl, reverse=True))
        else:
            temp_final[col_name] = temp_final[col].apply(lambda x: get_grade(x, temp_qtl))
        grade_cols.append(col_name)
    # 특정 변수 가중치
    if plus_dict:
        for plus_col in plus_dict.keys():
            temp_final[plus_col + '_점수'] = temp_final[plus_col + '_점수'] * plus_dict[plus_col]
    else:
        pass

    #     # 가격경쟁점수
    #     # 현재는 아직 확인 전이라 -1을 결측값으로 처리
    #     temp_final['11st_가격'] = temp_final['11st_가격'].apply(lambda x: np.nan if x==-1 else x)
    #     temp_final['Coupang_가격'] = temp_final['Coupang_가격'].apply(lambda x: np.nan if x==-1 else x)
    #     temp_final['Iherb_가격'] = temp_final['Iherb_가격'].apply(lambda x: np.nan if x==-1 else x)
    #     temp_final['Ople_가격'] = temp_final['Ople_가격'].apply(lambda x: np.nan if x==-1 else x)
    #     # 가격경쟁점수
    #     temp_final['가격점수'] = 0
    #     # 1. 쿠팡과 비교
    #     temp_final['가격점수'] = temp_final['가격점수'] + temp_final.apply(lambda x: 0 if np.isnan(x['Coupang_가격']) else 2 if x['몬짐가격']<=x['Coupang_가격'] else -2, axis=1)
    #     # 2. 타경쟁사와 비교
    #     temp_final['쿠팡외경쟁사최저'] = temp_final[['11st_가격', 'Iherb_가격', 'Ople_가격']].min(axis=1)
    #     temp_final['가격점수'] = temp_final['가격점수'] + temp_final.apply(lambda x: 0 if np.isnan(x['쿠팡외경쟁사최저']) \
    #                                                                else 1 if x['몬짐가격']<=x['쿠팡외경쟁사최저'] \
    #                                                                else -1, axis=1)
    #     # 3. 독점
    #     temp_final['쿠팡포함최저'] = temp_final[['11st_가격', 'Coupang_가격', 'Iherb_가격', 'Ople_가격']].min(axis=1) # 전체가 nan인지 apply lambda 내에서 확인하기 위해
    #     temp_final['가격점수'] = temp_final.apply(lambda x: 4 if np.isnan(x['쿠팡포함최저']) else x['가격점수'], axis=1)
    #     temp_final.drop(['쿠팡외경쟁사최저', '쿠팡포함최저'], axis=1, inplace=True)
    # #     temp_final['가격점수'] = 0
    # #     # 1. 쿠팡과 비교
    # #     temp_final['가격점수'] = temp_final['가격점수'] + temp_final.apply(lambda x: 0 if x['쿠팡가격']==-1 else 2 if x['몬짐가격']<=x['쿠팡가격'] else -2, axis=1)
    # #     # 2. 타경쟁사와 비교
    # #     temp_final['가격점수'] = temp_final['가격점수'] + temp_final.apply(lambda x: 0 if x['경쟁사최저가격']==-1 else 1 if x['몬짐가격']<=x['경쟁사최저가격'] else -1, axis=1)
    # #     # 3. 독점
    # #     temp_final['가격점수'] = temp_final.apply(lambda x: 4 if (x['쿠팡가격']==-1)&(x['경쟁사최저가격']==-1) else x['가격점수'], axis=1)
    #     grade_cols.append('가격점수')

    # 점수 합산
    temp_final['최종점수'] = temp_final[grade_cols].sum(axis=1)

    # penalty
    if minus_dict == None:
        pass
    else:
        for minus_col in minus_dict.keys():
            temp_final['최종점수'] = temp_final['최종점수'] - (temp_final[minus_col] * minus_dict[minus_col])

    # 컬럼정렬
    org_cols = temp_final.columns.tolist()
    score_cols = ['최종점수'] + grade_cols + list(minus_dict.keys())
    prod_cols = ['brand', '상품명']
    rest_cols = [col for col in org_cols if col not in score_cols + prod_cols]
    tot_cols = prod_cols + score_cols + rest_cols
    temp_final = temp_final[tot_cols]

    # benchmark 별도로 떼고 저장
    bchmrk_raw = pd.DataFrame(temp_final.loc[0]).T  # loc index 0 means 'benchmark'
    temp_final.drop(0, inplace=True)  # loc index 0 means 'benchmark'
    temp_final.sort_values(by='최종점수', ascending=False, inplace=True)
    temp_final = pd.concat([bchmrk_raw, temp_final])

    # 원본점수는 원상복구
    temp_final[rest_cols] = df[rest_cols]

    temp_final.drop_duplicates(subset='product_id', inplace=True)

    if head_num == None:
        temp_final = temp_final
    else:
        temp_final = temp_final.head(head_num + 1)

    temp_final.index = ['benchmark'] + list(range(1, len(temp_final)))

    #     # file save
    #     today_str = datetime.today().strftime('%Y%m%d')
    #     if filename_prefix==None:
    #         file_name = '상품랭킹_' + today_str + '.xlsx'
    #     else:
    #         file_name = filename_prefix + '_상품랭킹_' + today_str + '.xlsx'
    #     temp_final.to_excel(file_name)

    return temp_final


# 랭킹 최종 함수 업데이트
def product_ranking(
        anly_data, review_data, invt_data, final_bcd_stock, bcd_prod_df, brand_data, main_data, search_data,
        plus_cols, minus_dict, plus_dict, price_compare, exposed_data,
        filename_prefix=None,
        today='latest',
        latest_Ndays=7,
        compare_Ndays=28,
        atleast_avgqty=0,
        atleast_Nqty=6,
        #     in_stock=True,
        min_turnover=21,
        neg_review_rating=3,
        target_period=180,
        min_repurchase=30,
        min_repurchase_user=10,
        head_num=None,
        this_year_Ndays=30,
        this_quat_Ndays=20
):
    ### 판매량 상승률 관련
    # 장기상승률 추가
    no_year_ago = anly_data['purchased_at'].max()
    a_year_ago = anly_data['purchased_at'].max() - relativedelta(months=12)
    two_years_ago = anly_data['purchased_at'].max() - relativedelta(months=24)

    this_year = anly_data[(anly_data['purchased_at'] > a_year_ago) & (anly_data['purchased_at'] <= no_year_ago)]
    last_year = anly_data[(anly_data['purchased_at'] > two_years_ago) & (anly_data['purchased_at'] <= a_year_ago)]

    # 적어도 올해 this_year_Ndays일 이상은 팔린 상품들에 대해서 값 연산
    this_year_cnt = this_year.groupby('product_id')['purchased_at'].nunique().reset_index()
    over_Ndays = this_year_cnt[this_year_cnt['purchased_at'] >= this_year_Ndays]['product_id'].unique()

    # 일평균 판매량을 구하려면, 상품별 일단위 합을 우선 구한 다음, 평균내야 함
    this_year = this_year.groupby(['product_id', 'purchased_at'])['product_qty'].sum().reset_index()
    this_year = this_year.groupby('product_id')['product_qty'].mean().reset_index()
    this_year.rename(columns={'product_qty': 'this_year_qty'}, inplace=True)
    this_year = this_year[this_year['product_id'].isin(over_Ndays)]
    last_year = last_year.groupby(['product_id', 'purchased_at'])['product_qty'].sum().reset_index()
    last_year = last_year.groupby('product_id')['product_qty'].mean().reset_index()
    last_year.rename(columns={'product_qty': 'last_year_qty'}, inplace=True)

    ltg_data = pd.merge(this_year, last_year, on='product_id', how='left')

    # 작년 판매기록 없는 상품들은 가장 과거 한달 평균판매량 대비로 변경
    last_none = ltg_data[ltg_data['last_year_qty'].isnull()]['product_id']
    # 가장 과거 한달이 6개월 내이면, 분기상승률(과거3개월 대비 최근3개월)에서 어느정도 추세가 나올 것이기 때문에 6개월 이내인 상품은 0으로 제외
    first_sold = anly_data[anly_data['product_id'].isin(last_none)].groupby('product_id')[
        'purchased_at'].min().reset_index()
    first_sold = first_sold[
        first_sold['purchased_at'].apply(lambda x: (anly_data['purchased_at'].max() - x).days >= 180)]
    first_sold.rename(columns={'purchased_at': 'first_sold_at'}, inplace=True)
    # last_year을 대신할 가장 과거 한달 테이블
    altn_lastyear = anly_data.copy()
    altn_lastyear = pd.merge(altn_lastyear, first_sold, on='product_id', how='inner')
    altn_lastyear = altn_lastyear[(altn_lastyear['purchased_at'] >= altn_lastyear['first_sold_at']) & (
            altn_lastyear['purchased_at'] <= altn_lastyear['first_sold_at'] + timedelta(days=30))]
    altn_lastyear = altn_lastyear.groupby(['product_id', 'purchased_at'])['product_qty'].sum().reset_index()
    altn_lastyear = altn_lastyear.groupby('product_id')['product_qty'].mean().reset_index()
    altn_lastyear.rename(columns={'product_qty': 'altn_lastyear_qty'}, inplace=True)

    # 기존 ltg_data와 merge
    ltg_data = pd.merge(ltg_data, altn_lastyear, on='product_id', how='left')
    ltg_data['last_year_qty'] = ltg_data['last_year_qty'].fillna(ltg_data['altn_lastyear_qty'])
    ltg_data['YOY_increase'] = np.round(ltg_data['this_year_qty'] / ltg_data['last_year_qty'], 4)
    ltg_data['YOY_increase'] = ltg_data['YOY_increase'].fillna(
        0)  # 가장 과거가 6개월도 안되는 상품은 분기상승률에서 어느정도 반영되기 때문에 연단위상승률을 0으로 채우기

    # 분기상승률 추가
    no_quat_ago = anly_data['purchased_at'].max()
    a_quat_ago = anly_data['purchased_at'].max() - relativedelta(months=3)
    two_quats_ago = anly_data['purchased_at'].max() - relativedelta(months=6)

    this_quat = anly_data[(anly_data['purchased_at'] > a_quat_ago) & (anly_data['purchased_at'] <= no_quat_ago)]
    last_quat = anly_data[(anly_data['purchased_at'] > two_quats_ago) & (anly_data['purchased_at'] <= a_quat_ago)]

    # 적어도 이번달 this_quat_Ndays일 이상은 팔린 상품들에 대해서 값 연산
    this_quat_cnt = this_quat.groupby('product_id')['purchased_at'].nunique().reset_index()
    over_Ndays = this_quat_cnt[this_quat_cnt['purchased_at'] >= this_quat_Ndays]['product_id'].unique()

    this_quat = this_quat.groupby(['product_id', 'purchased_at'])['product_qty'].sum().reset_index()
    this_quat = this_quat.groupby('product_id')['product_qty'].mean().reset_index()
    this_quat.rename(columns={'product_qty': 'this_quat_qty'}, inplace=True)
    this_quat = this_quat[this_quat['product_id'].isin(over_Ndays)]
    last_quat = last_quat.groupby(['product_id', 'purchased_at'])['product_qty'].sum().reset_index()
    last_quat = last_quat.groupby('product_id')['product_qty'].mean().reset_index()
    last_quat.rename(columns={'product_qty': 'last_quat_qty'}, inplace=True)

    quatly_ltg_data = pd.merge(this_quat, last_quat, on='product_id', how='left')
    quatly_ltg_data['QOQ_increase'] = np.round(quatly_ltg_data['this_quat_qty'] / quatly_ltg_data['last_quat_qty'], 4)
    quatly_ltg_data['QOQ_increase'] = quatly_ltg_data['QOQ_increase'].fillna(
        0)  # 분기상승률은 간격이 너무 짧기 때문에 연단위상승률처럼 굳이 대조군이 없을 때 채우지 않음.

    # → 여기서부터 원래 급상승
    if today == 'latest':
        today = anly_data.purchased_at.max()
    else:
        today = datetime.strptime(today, '%Y-%m-%d')

    latest_df = anly_data[anly_data.purchased_at.between(today - timedelta(days=latest_Ndays), today)]
    compare_df = anly_data[
        anly_data.purchased_at.between(today - timedelta(days=latest_Ndays) - timedelta(days=compare_Ndays),
                                       today - timedelta(days=latest_Ndays))]

    # 0판매 복구시켜줄 재고 확인 테이블
    stock_left = invt_data[invt_data.amount != 0]  # 재고가 0이 아닌 테이블
    stock_left = pd.merge(stock_left, bcd_prod_df[['barcode', 'product_id']], on='barcode', how='left')
    stock_left = stock_left.groupby(['product_id', 'sys_time'])['amount'].sum().reset_index()
    stock_left.rename(columns={'sys_time': 'purchased_at'}, inplace=True)
    stock_left.drop('amount', axis=1, inplace=True)

    # 그냥 groupby mean을 하면 한 거래단위의 평균값이 나오기 때문에 거의 한 두개. 따라서 일단위 합계를 구한 뒤 평균을 내야 함
    latest_df = latest_df.groupby(['product_id', 'purchased_at'])['product_qty'].sum().reset_index()
    # latest_Ndays 기간동안 적어도 N개 이상 팔린 상품들에 대해서만 필터링
    atleast_df = latest_df.groupby('product_id')['product_qty'].sum()
    atleast_list = atleast_df[atleast_df > atleast_Nqty].index.tolist()
    latest_df = latest_df[latest_df.product_id.isin(atleast_list)]
    latest_df = pd.merge(latest_df, stock_left, on=['product_id', 'purchased_at'], how='outer')
    latest_df.product_qty.fillna(0, inplace=True)
    latest_df = latest_df[latest_df.purchased_at.between(today - timedelta(days=latest_Ndays), today)]
    latest_df = latest_df.groupby('product_id')['product_qty'].mean().reset_index()
    latest_df.rename(columns={'product_qty': 'latest_qty'}, inplace=True)

    compare_df = compare_df.groupby(['product_id', 'purchased_at'])['product_qty'].sum().reset_index()
    compare_df = pd.merge(compare_df, stock_left, on=['product_id', 'purchased_at'], how='outer')
    compare_df.product_qty.fillna(0, inplace=True)
    compare_df = compare_df[
        compare_df.purchased_at.between(today - timedelta(days=latest_Ndays) - timedelta(days=compare_Ndays),
                                        today - timedelta(days=latest_Ndays))]
    compare_df = compare_df.groupby('product_id')['product_qty'].mean().reset_index()
    compare_df.rename(columns={'product_qty': 'compare_qty'}, inplace=True)
    compare_df.compare_qty = compare_df.compare_qty.apply(lambda x: 1 if x < 1 else x)  # 분모가 1보다 작으면 1로 보정

    merged_df = pd.merge(latest_df, compare_df, how='left')  # 대조군에 없어서 분모가 0이되면 무한대가 되니 제외
    merged_df.compare_qty.fillna(1, inplace=True)  # 과거에 판매가 없었다면 0인데, 분모가 될 수 없으니 1로 보정
    merged_df['increase_rate'] = merged_df['latest_qty'] / merged_df['compare_qty']
    merged_df.sort_values(by='increase_rate', ascending=False, inplace=True)
    merged_df = merged_df[merged_df.latest_qty >= atleast_avgqty]

    merged_df = pd.merge(merged_df, anly_data[['product_id', 'product_name_kor']].drop_duplicates(), how='left')
    merged_df = pd.merge(merged_df, brand_data, on='product_id', how='left')
    merged_df = pd.merge(merged_df, ltg_data, on='product_id', how='left')
    merged_df = pd.merge(merged_df, quatly_ltg_data, on='product_id', how='left')
    merged_df = merged_df[~merged_df['brand'].isin(filtered_brand)]
    #     merged_df = merged_df[['product_id', 'product_name_kor', 'brand', 'latest_qty', 'compare_qty', 'increase_rate']]

    # 자릿수 줄이기
    merged_df['latest_qty'] = np.round(merged_df['latest_qty'], 2)
    merged_df['compare_qty'] = np.round(merged_df['compare_qty'], 2)
    merged_df['increase_rate'] = np.round(merged_df['increase_rate'], 2)

    merged_df.rename(columns={
        'product_name_kor': '상품명',
        'latest_qty': '최근' + str(latest_Ndays) + '일 일평균판매량',
        'compare_qty': '과거' + str(compare_Ndays) + '일 일평균판매량',
        'increase_rate': '일주일상승률',
        'YOY_increase': '연상승률',
        'QOQ_increase': '분기상승률'
    }, inplace=True)
    merged_df.drop_duplicates(subset='product_id', inplace=True)

    ### 재구매 관련
    temp_anly_data = anly_data[anly_data.purchased_at.between(today - timedelta(days=target_period), today)]

    # 1) 구매건수기반의 재구매 상품(재구매건수/전체구매건수)
    tot_pchs = temp_anly_data.groupby('product_id')['user_id'].count().reset_index()
    tot_pchs.rename(columns={'user_id': 'total_purchase'}, inplace=True)
    fst_pchs = temp_anly_data.groupby('product_id')['user_id'].nunique().reset_index()
    fst_pchs.rename(columns={'user_id': 'first_purchase'}, inplace=True)
    repurchase_rate = pd.merge(tot_pchs, fst_pchs, on='product_id', how='inner')
    repurchase_rate['repurchase'] = repurchase_rate.total_purchase - repurchase_rate.first_purchase
    repurchase_rate['repurchase_rate'] = repurchase_rate.repurchase / repurchase_rate.total_purchase
    repurchase_rate.sort_values(by='repurchase_rate', ascending=False, inplace=True)
    repurchase_rate = repurchase_rate[['product_id', 'repurchase', 'total_purchase', 'repurchase_rate']]

    # 자릿수 줄이기
    repurchase_rate['repurchase_rate'] = np.round(repurchase_rate['repurchase_rate'], 2)
    repurchase_rate.rename(columns={
        'total_purchase': '전체구매건수',
        'repurchase': '재구매건수',
        'repurchase_rate': '재구매건수비율'
    }, inplace=True)

    # 2) 구매유저수기반의 재구매 상품(재구매한유저수/전체구매유저수)
    temp_anly_data = temp_anly_data.sort_values(by=['user_id', 'purchased_at'])
    first_order = temp_anly_data.drop_duplicates(subset=['user_id', 'product_id'], keep='first')  # 유저별, 상품별로 첫 구매 기록
    retemp_anly_data = temp_anly_data.drop(first_order.index, axis=0)  # 첫구매 제외한 재구매 데이터들만
    # 재구매건수 중 unique한 유저의 비율(즉, 1에 가까울수록 general한 구매, 0에 가까울수록 maniac한 구매)
    retot_pchs = retemp_anly_data.groupby('product_id')['user_id'].count().reset_index()
    retot_pchs.rename(columns={'user_id': 'total_repurchase'}, inplace=True)
    unique_users = retemp_anly_data.groupby('product_id')['user_id'].nunique().reset_index()
    unique_users.rename(columns={'user_id': 'unique_users'}, inplace=True)
    general_rate = pd.merge(retot_pchs, unique_users, on='product_id', how='inner')
    general_rate['general_repurchase_rate'] = general_rate.unique_users / general_rate.total_repurchase
    general_rate = general_rate[general_rate.total_repurchase >= min_repurchase]
    general_rate = general_rate[general_rate.unique_users >= min_repurchase_user]
    general_rate.sort_values(by='general_repurchase_rate', ascending=False, inplace=True)
    general_rate = pd.merge(general_rate, temp_anly_data[['product_id', 'category_M', 'category_S']].drop_duplicates(),
                            how='left')
    general_rate = general_rate[
        ['product_id', 'category_M', 'category_S', 'unique_users', 'total_repurchase', 'general_repurchase_rate']]

    # 자릿수 줄이기
    general_rate['general_repurchase_rate'] = np.round(general_rate['general_repurchase_rate'], 2)
    general_rate.rename(columns={
        'category_M': '메인카테고리',
        'category_S': '서브카테고리',
        'unique_users': '재구매유저수',
        'total_repurchase': '재구매건수',
        'general_repurchase_rate': '재구매건수중유저비율'
    }, inplace=True)

    # 두 재구매율 함께 고려
    repurchase_df = pd.merge(repurchase_rate, general_rate)
    repurchase_df['재구매지수'] = repurchase_df['재구매건수비율'] * repurchase_df['재구매건수중유저비율']
    repurchase_df.drop_duplicates(subset='product_id', inplace=True)

    ### review_data
    review_data = review_data[
        review_data.post_time >= (today - timedelta(days=latest_Ndays + 1))]  # 실행한날 24시가 지나지 않았으니 그냥 하루 추가
    review_data.rename(columns={'object_id': 'product_id'}, inplace=True)
    review_data['message_length'] = review_data.message.apply(lambda x: len(x))
    review_length = review_data.groupby('product_id')['message_length'].mean().reset_index()
    review_cnt = review_data.groupby('product_id')['post_id'].nunique().reset_index()
    review_rating = review_data.groupby('product_id')['rating_value'].mean().reset_index()
    review_neg = review_data.groupby('product_id')['rating_value'].apply(
        lambda x: x[x <= neg_review_rating].count()).reset_index()
    review_neg.rename(columns={'rating_value': 'neg_review'}, inplace=True)
    review_neg_raw = review_data[review_data.rating_value <= neg_review_rating].sort_values(by='rating_value')[
        ['product_id', 'message']].drop_duplicates(subset='product_id')  # 실제 부정리뷰 1건
    review_df = pd.merge(review_rating, review_cnt, on='product_id')
    review_df = pd.merge(review_df, review_length, on='product_id')
    review_df = pd.merge(review_df, review_neg, on='product_id', how='left')
    review_df.neg_review.fillna(0, inplace=True)
    review_df = pd.merge(review_df, review_neg_raw, on='product_id', how='left')
    review_df.rename(columns={
        'rating_value': '평균평점',
        'post_id': '리뷰개수',
        'message_length': '평균리뷰길이',
        'neg_review': '부정리뷰(' + str(neg_review_rating) + '점이하)',
        'message': '부정리뷰사례'
    }, inplace=True)
    review_df['평균평점'] = np.round(review_df['평균평점'], 2)
    review_df['평균리뷰길이'] = np.round(review_df['평균리뷰길이'], 2)

    ### invt_data
    # @@@ OOS의 경우 product_id - barcode가 일대다대응이기 때문에 일단 통합으로 진행
    temp_invt_data = invt_data[
        invt_data.sys_time >= (today - timedelta(days=latest_Ndays - 1))]  # anly_data의 날짜 기준과 약간의 차이가 있음
    temp_invt_data = pd.merge(temp_invt_data, anly_data[['product_id', 'barcode']].drop_duplicates(), on='barcode',
                              how='left')  # 바코드에 product_id 붙여주기
    temp_invt_data = temp_invt_data[temp_invt_data.product_id.notnull()]
    temp_invt_data.product_id = temp_invt_data.product_id.astype('int')
    # 일단 product_id 기준으로 재고 통합
    temp_invt_data = temp_invt_data.groupby(['product_id', 'sys_time'])['amount'].sum().reset_index()
    invtday_count = temp_invt_data.groupby('product_id')['sys_time'].nunique().reset_index()
    oosday_count = temp_invt_data.groupby('product_id')['amount'].apply(lambda x: x[x == 0].count()).reset_index()
    invt_df = pd.merge(invtday_count, oosday_count, on='product_id', how='left')
    invt_df['oos_rate'] = invt_df.amount / invt_df.sys_time
    invt_df.rename(columns={
        'sys_time': '재고관리총일수',
        'amount': '결품일수',
        'oos_rate': 'OOS_RATE',
    }, inplace=True)

    ### search_data
    search_data = search_data[search_data.sys_time.between(today - timedelta(days=latest_Ndays), today)]
    searched_df = search_data.groupby('product_id')['idx'].nunique().reset_index()
    searched_df.rename(columns={'idx': '검색유입수'}, inplace=True)

    ### 최종 취합 및 정렬
    # 지난 일주일간 적어도 판매가 있었다면 급상승 및 재구매 결과가 나오기 때문에 inner join. 그 외 리뷰 및 한국제품은 재고테이블에 정보가 없기 때문에 left join
    final_df = pd.merge(merged_df, repurchase_df, on='product_id', how='inner')
    final_df = pd.merge(final_df, review_df, on='product_id', how='left')
    final_df = pd.merge(final_df, invt_df, on='product_id', how='left')
    final_df = pd.merge(final_df, searched_df, on='product_id', how='left')

    final_df['연상승률'].fillna(0, inplace=True)  # 최소판매회수 기준에서 빠졌으면 장기상승률 계산 하지 않고 0으로 취급
    final_df['분기상승률'].fillna(0, inplace=True)  # 최소판매회수 기준에서 빠졌으면 장기상승률 계산 하지 않고 0으로 취급
    final_df['서브카테고리'].fillna('-', inplace=True)
    final_df['평균평점'].fillna(final_df['평균평점'].mean(), inplace=True)
    final_df['리뷰개수'].fillna(0, inplace=True)
    final_df['평균리뷰길이'].fillna(0, inplace=True)
    final_df['검색유입수'].fillna(0, inplace=True)
    final_df['부정리뷰(' + str(neg_review_rating) + '점이하)'].fillna(0, inplace=True)
    final_df['부정리뷰사례'].fillna('-', inplace=True)
    # final_df.fillna(final_df.mean(), inplace=True)
    final_df = pd.merge(final_df, final_bcd_stock, how='outer')

    #     if in_stock==True:
    #         final_df = final_df[final_df['1st판매옵션현재고']>0]
    #     else:
    #         pass

    # 재고전환일수 필터링
    cardillo_stock = final_df[final_df['product_id'] == 314]  # 카딜로는 제고 고려 제외
    non_stock_df = final_df[final_df['1st판매옵션재고전환일수'].isnull()]  # 재고정보 없는 한국제품들
    #     even_sell_option_df = final_df[(final_df['1st판매옵션비율'] + final_df[
    #         '2nd판매옵션비율']) <= 0.5]  # 옵션이 골고루 판매되는 상품은 1, 2번째 옵션 재고가 적더라도 그 이후 재고가 많으면 보여질 수 있음
    final_df = final_df[(final_df['1st판매옵션재고전환일수'] + final_df['2nd판매옵션재고전환일수']) >= min_turnover]
    #     final_df = final_df[(final_df['1st판매옵션재고전환일수']>=min_turnover)|(final_df['2nd판매옵션재고전환일수']>=min_turnover)]
    final_df = pd.concat([final_df, non_stock_df, cardillo_stock], ignore_index=True)
    final_df.reset_index(drop=True, inplace=True)

    # 일단 raw data는 판매량 기준으로 정렬. 상품 순위는 각 스토리에 맞게 바뀔 예정
    final_df.sort_values(by='최근' + str(latest_Ndays) + '일 일평균판매량', ascending=False, inplace=True)
    final_df = final_df[
        final_df['상품명'].notnull()]  # 상품명 확인 안되는 제품 제외(마스터에서는 확인되나, 실제로 판매가 일어나지 않은 상품들에 해당. 재고정보만 확인되는 케이스)
    final_df.reset_index(drop=True, inplace=True)
    final_df = pd.merge(final_df, price_compare, on='product_id', how='left')

    # benchmark
    benchmark = pd.DataFrame(final_df.mean()).T
    benchmark.index = ['benchmark']
    benchmark['product_id'] = np.nan
    benchmark['1st판매옵션바코드'] = np.nan
    benchmark['2nd판매옵션바코드'] = np.nan
    benchmark = np.round(benchmark, 2)
    final_df = pd.concat([benchmark, final_df], sort=False)

    # 컬럼순서 변경
    final_df = final_df[[
        'brand', '상품명',
        '몬짐가격', '11st_가격', 'Coupang_가격', 'Iherb_가격', 'Ople_가격',
        '최근' + str(latest_Ndays) + '일 일평균판매량', '과거' + str(compare_Ndays) + '일 일평균판매량',
        '일주일상승률', '분기상승률', '연상승률',
        '재구매지수', '재구매건수', '전체구매건수', '재구매건수비율', '재구매유저수', '재구매건수중유저비율',
        '평균평점', '리뷰개수', '평균리뷰길이', '부정리뷰(' + str(neg_review_rating) + '점이하)', '부정리뷰사례',
        '결품일수', 'OOS_RATE', '검색유입수',
        'product_id', '메인카테고리', '서브카테고리',
        '1st판매옵션바코드', '1st판매옵션명', '1st판매옵션비율', '1st판매옵션현재고', '1st판매옵션재고전환일수',
        '2nd판매옵션바코드', '2nd판매옵션명', '2nd판매옵션비율', '2nd판매옵션현재고', '2nd판매옵션재고전환일수',

    ]]

    final_df = pd.merge(final_df, exposed_data, on='product_id', how='left')
    final_df = get_final_score(final_df, plus_cols, minus_dict, plus_dict, filename_prefix, head_num)
    return final_df


# 메인추천카테고리 상품 중복 제거 (우선순위: 랭킹 > 재구매 > 판매급상승)
whole_head_num = 50  # 메인카테고리 상품 중복 제거 작업 후 head number. 몇 개 뽑을 것인지
top_list_head_num = 15  # whole_head_num는 3개 각 테이블의 raw 결과시트용 순위이고, 이 순위는 노출 순위 시 앞선 우선순위 카테고리를 고려하기 위한 순위

# 1. 랭킹결과
plus_cols = ['최근7일 일평균판매량', '일주일상승률', '분기상승률', '연상승률',
             '재구매지수', '검색유입수']
minus_dict = {'부정리뷰(3점이하)': 2}  # 감점컬럼 및 배수
plus_dict = {'최근7일 일평균판매량': 3}  # 가중치컬럼 및 배수
ranking_result = product_ranking(anly_data, review_data, invt_data, final_bcd_stock, bcd_prod_df, brand_data, main_data,
                                 search_data,
                                 plus_cols, minus_dict, plus_dict, price_compare, exposed_data,
                                 min_turnover=21, head_num=whole_head_num)
# 랭킹에 등장한 상품 리스트(이 때는 전체 50개를 고려하는게 아니라, 노출되는 top_list_head_num만 고려)
rank_prods = ranking_result.drop('benchmark', axis=0).head(top_list_head_num)['product_id'].unique()

# 2. 재구매결과
repurchase_result = repurchase_prod(anly_data, final_bcd_stock, brand_data, main_data, exposed_data, price_compare,
                                    target_period=90, head_num=None)
repurchase_result = repurchase_result[~repurchase_result['product_id'].isin(rank_prods)].head(
    whole_head_num + 1)  # 랭킹 리스트에 등장한 상품 제외
repurchase_result.index = ['benchmark'] + list(range(1, len(repurchase_result)))  # 인덱스 수정
rep_prods = repurchase_result.drop('benchmark', axis=0)['product_id'].unique()  # 재구매에 등장한 상품 리스트

# 3. 급상승결과-1(카딜로 처리로 인한 구분 처리)
increase_result = rap_increase_prod(anly_data, final_bcd_stock, invt_data, bcd_prod_df, brand_data, main_data,
                                    exposed_data, price_compare,
                                    head_num=None)

### 최종결과물: ranking_result, repurchase_result, increase_result

# file save
# today = datetime.today().strftime('%Y%m%d')
# path = '급상승&재구매&랭킹_{}.xlsx'.format(today)
# writer = pd.ExcelWriter(path, engine='xlsxwriter')
#
# ranking_result.to_excel(writer, sheet_name='랭킹결과')
# repurchase_result.to_excel(writer, sheet_name='재구매결과')
# increase_result.to_excel(writer, sheet_name='급상승결과')
#
# writer.save()
# writer.close()


# DB upload table
ranking_top20 = pd.DataFrame(ranking_result['product_id'][1:top_list_head_num + 1].astype('int'))
ranking_top20['date'] = datetime.today()
ranking_top20['main_rec_category'] = '주간 랭킹'
ranking_top20['location'] = list(range(1, len(ranking_top20)+1))

repurchase_top20 = pd.DataFrame(repurchase_result['product_id'][1:top_list_head_num + 1].astype('int'))
repurchase_top20['date'] = datetime.today()
repurchase_top20['main_rec_category'] = '재구매율 높은'
repurchase_top20['location'] = list(range(1, len(repurchase_top20)+1))

increase_top20 = pd.DataFrame(increase_result['product_id'][1:top_list_head_num + 1].astype('int'))
increase_top20['date'] = datetime.today()
increase_top20['main_rec_category'] = '판매 급상승'
increase_top20['location'] = list(range(1, len(increase_top20)+1))

# 카딜로벨트 필터링 작업 추가
# 2. 노출(상위 top_list_head_num등)되는 카테고리 하나라도 있으면 그대로 두고, 없으면 판매급상승 1위로.
exp_1 = ranking_result['product_id'][1:top_list_head_num + 1].astype('int').tolist()
exp_2 = repurchase_result['product_id'][1:top_list_head_num + 1].astype('int').tolist()
exp_3 = increase_result['product_id'][1:top_list_head_num + 1].astype('int').tolist()
exp_1.extend(exp_2)
exp_1.extend(exp_3)
if 314 in exp_1:  # 카딜로벨트가 3개 카테고리의 상위 15등 내에 있으면 패스
    pass
elif ((cardillo_invt['amount'] > 0).sum() >= 1) & (314 not in exp_1):  # 재고가 있는데 15등 안에 안나왔을 경우, 판매급상승 1위를 카딜로벨트로
    increase_1st = pd.DataFrame(increase_top20.loc[1].copy()).T
    increase_1st['product_id'] = 314
    increase_last = increase_top20[:-1]
    increase_last['location'] = increase_last['location'] + 1
    increase_top20 = pd.concat([increase_1st, increase_last])
    # increase_result 테이블에서 카딜로 살려내기
    increase_benchmark = increase_result[increase_result.index == 'benchmark']
    increase_cardilo = increase_result[increase_result['product_id'] == 314]
    increase_lefts = increase_result[(increase_result.index != 'benchmark') & (increase_result['product_id'] != 314)]
    increase_result = pd.concat([increase_benchmark, increase_cardilo, increase_lefts])
else:  # 재고가 없어서 안나왔으면 그냥 패스
    pass
# raw data를 저장하는 각 카테고리의 테이블에서도 카딜로벨트를 복구할 수는 없음 
# 산출 시 이미 head_num으로 필터링 후 작업하기 때문에 항상 head_num을 None으로 하지 않으면 불가능함. 
# 만약 head_num을 None으로 한다면, 각 테이블의 행수는 산출되는 상품수만큼 늘어나게 될 것 
# → 복구처리 완료

# 3. 급상승결과-2(카딜로 처리로 인한 구분 처리)
rank_rep_prods = np.append(rank_prods, rep_prods)  # 랭킹 & 재구매 리스트에 등장한 상품 제외
increase_result = increase_result[~increase_result['product_id'].isin(rank_rep_prods)].head(whole_head_num + 1)
increase_result.index = ['benchmark'] + list(range(1, len(increase_result)))  # 인덱스 수정

cscart_main_recommend_product_list = pd.concat([ranking_top20, repurchase_top20, increase_top20], ignore_index=True)
cscart_main_recommend_product_list.column = ['PRODUCT_ID', 'DATE', 'MAIN_REC_CATEGORY', 'LOCATION']

cscart_main_recommend_ranking = ranking_result.copy()
cscart_main_recommend_ranking.drop('benchmark', axis=0, inplace=True)
cscart_main_recommend_ranking['date'] = datetime.today()
cscart_main_recommend_ranking['rank'] = list(range(1, len(cscart_main_recommend_ranking) + 1))
cscart_main_recommend_ranking.columns = [
    'BRAND', 'PRODUCT_NAME_KOR', 'SCORE_TOTAL', 'SCORE_AVG_QTY', 'SCORE_GRWRATE_WEEK',
    'SCORE_GRWRATE_QUT', 'SCORE_GRWRATE_YEAR', 'SCORE_REPCHS', 'SCORE_SEARCH', 'SCORE_NEG_RVW',
    'PRICE_MZ', 'PRICE_11ST', 'PRICE_COUPANG', 'PRICE_IHERB', 'PRICE_OPLE', 'LAST_AVG_QTY',
    'PAST_AVG_QTY', 'GRWRATE_WEEK', 'GRWRATE_QUT', 'GRWRATE_YEAR', 'REPCHS_INDEX', 'REPCHS_CNT',
    'PCHS_CNT', 'REPCHS_CNT_RATE', 'REPCHS_USER', 'REPCHS_USER_RATE', 'AVG_RATING', 'RVW_CNT',
    'RVW_LENGTH', 'NEG_RVW_CASE', 'OOS_DAYS', 'OOS_RATE', 'SEARCH_NUM', 'PRODUCT_ID',
    'CATEGORY_M', 'CATEGORY_S', 'OPTION_BARCODE_1ST', 'OPTION_NAME_1ST', 'OPTION_SALE_RATE_1ST',
    'OPTION_STCK_1ST', 'OPTION_DSI_1ST', 'OPTION_BARCODE_2ND', 'OPTION_NAME_2ND', 'OPTION_SALE_RATE_2ND',
    'OPTION_STCK_2ND', 'OPTION_DSI_2ND', 'MAIN_REC_LAST_CATEGORY', 'MAIN_REC_START_DATE',
    'MAIN_REC_GRWRATE', 'DATE', 'RANK_NUM'
]

cscart_main_recommend_repurchase = repurchase_result.copy()
cscart_main_recommend_repurchase.drop('benchmark', axis=0, inplace=True)
cscart_main_recommend_repurchase['date'] = datetime.today()
cscart_main_recommend_repurchase['rank'] = list(range(1, len(cscart_main_recommend_repurchase) + 1))
cscart_main_recommend_repurchase.columns = [
    'BRAND', 'PRODUCT_NAME_KOR', 'SCORE_TOTAL', 'SCORE_REPCHS_CNT_RATE', 'SCORE_REPCHS_USER_RATE',
    'SCORE_AVG_QTY', 'REPCHS_INDEX', 'PRICE_MZ', 'REPCHS_CNT_RATE', 'REPCHS_USER_RATE', 'REPCHS_CNT', 'PCHS_CNT',
    'REPCHS_USER',
    'LAST_AVG_QTY', 'OPTION_STCK_1ST', 'OPTION_DSI_1ST', 'OPTION_SALE_RATE_1ST', 'OPTION_STCK_2ND',
    'OPTION_DSI_2ND', 'OPTION_SALE_RATE_2ND', 'PRODUCT_ID', 'OPTION_BARCODE_1ST', 'OPTION_BARCODE_2ND',
    'PRICE_11ST', 'PRICE_COUPANG', 'PRICE_IHERB', 'PRICE_OPLE',
    'CATEGORY_M', 'CATEGORY_S', 'OPTION_NAME_1ST', 'OPTION_NAME_2ND', 'MAIN_REC_LAST_CATEGORY',
    'MAIN_REC_START_DATE', 'MAIN_REC_GRWRATE', 'DATE', 'RANK_NUM'
]

cscart_main_recommend_increase = increase_result.copy()
cscart_main_recommend_increase.drop('benchmark', axis=0, inplace=True)
cscart_main_recommend_increase['date'] = datetime.today()
cscart_main_recommend_increase['rank'] = list(range(1, len(cscart_main_recommend_increase) + 1))
cscart_main_recommend_increase.columns = [
    'BRAND', 'PRODUCT_NAME_KOR', 'GRWRATE', 'LAST_AVG_QTY', 'PAST_AVG_QTY', 'PRICE_MZ',
    'PRICE_11ST', 'PRICE_COUPANG', 'PRICE_IHERB', 'PRICE_OPLE', 'LT_GRWRATE', 'OPTION_NAME_1ST',
    'OPTION_STCK_1ST', 'OPTION_DSI_1ST', 'OPTION_SALE_RATE_1ST', 'OPTION_NAME_2ND', 'OPTION_STCK_2ND',
    'OPTION_DSI_2ND', 'OPTION_SALE_RATE_2ND', 'PRODUCT_ID', 'OPTION_BARCODE_1ST', 'OPTION_BARCODE_2ND',
    'MAIN_REC_LAST_CATEGORY', 'MAIN_REC_START_DATE', 'MAIN_REC_GRWRATE', 'DATE', 'RANK_NUM'
]

cscart_main_recommend_ranking['REPCHS_INDEX'] = np.round(cscart_main_recommend_ranking['REPCHS_INDEX'], 4)
cscart_main_recommend_ranking['AVG_RATING'] = np.round(cscart_main_recommend_ranking['AVG_RATING'], 4)

cscart_main_recommend_repurchase['REPCHS_INDEX'] = np.round(cscart_main_recommend_repurchase['REPCHS_INDEX'], 4)

# file save
today = datetime.today().strftime('%Y%m%d')
path = '{}_결과.xlsx'.format(today)
writer = pd.ExcelWriter(path, engine='xlsxwriter')

cscart_main_recommend_product_list.to_excel(writer, sheet_name='상품리스트')
cscart_main_recommend_ranking.to_excel(writer, sheet_name='랭킹')
cscart_main_recommend_repurchase.to_excel(writer, sheet_name='재구매')
cscart_main_recommend_increase.to_excel(writer, sheet_name='급상승')

writer.save()
writer.close()
