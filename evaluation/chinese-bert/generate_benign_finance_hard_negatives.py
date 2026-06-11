from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONTRAST_JSONL = PROJECT_ROOT / "evaluation" / "finance_semantic_contrast_train_val.jsonl"
CONTRAST_CSV = PROJECT_ROOT / "evaluation" / "finance_semantic_contrast_train_val.csv"
CONTRAST_SUMMARY_CSV = PROJECT_ROOT / "evaluation" / "finance_semantic_contrast_summary.csv"

LEGACY_BENIGN_JSONL = PROJECT_ROOT / "evaluation" / "benign_finance_hard_negatives_train_val.jsonl"
LEGACY_BENIGN_CSV = PROJECT_ROOT / "evaluation" / "benign_finance_hard_negatives_train_val.csv"
LEGACY_BENIGN_SUMMARY_CSV = PROJECT_ROOT / "evaluation" / "benign_finance_hard_negatives_summary.csv"

PER_NORMAL_SCENARIO = 30
PER_FRAUD_SCENARIO = 24


def pick(items: List[str], index: int, offset: int = 0) -> str:
    return items[(index + offset) % len(items)]


def split_for(index: int) -> str:
    return "val" if index % 5 == 0 else "train"


def normal_memory_suffix(index: int) -> str:
    days = ["一号", "二号", "三号", "四号", "五号", "六号", "七号", "八号", "九号", "十号"]
    periods = ["早上", "中午", "下午", "晚上", "周末", "月初", "月底", "发工资后", "报销前", "开会后"]
    return f"\n甲：我记得那次大概是{pick(days, index)}{pick(periods, index, 3)}，所以印象比较深。\n乙：嗯，这种生活细节聊起来挺常见的。"


def fraud_pressure_suffix(index: int) -> str:
    tickets = ["A17", "B29", "C38", "D46", "E52", "F61", "G73", "H84", "J95", "K06"]
    windows = ["三分钟", "五分钟", "八分钟", "十分钟", "十五分钟", "这一通电话内"]
    return f"\n甲：您这个工单尾号{pick(tickets, index)}，处理窗口只剩{pick(windows, index, 2)}。\n乙：那我是不是现在就得按你说的做？"


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？；：,.!?;:（）()\[\]{}【】\"'“”‘’·…—\-_/\\|<>~`@#$%^&*+=]", "", text)
    return text.lower()


def normal_payment_app_discussion(i: int) -> Dict[str, str]:
    apps = ["微信支付", "支付宝", "花呗", "借呗", "云闪付", "银行App"]
    angles = ["方便", "安全性", "额度", "还款", "付款速度", "账单记录"]
    scenes = ["吃饭结账", "交水电费", "买车票", "给家里买东西", "学校缴费", "公司报销"]
    app = pick(apps, i)
    angle = pick(angles, i, 1)
    scene = pick(scenes, i, 2)
    templates = [
        f"甲：我现在基本都用{app}，主要是{scene}的时候比较省事。\n乙：对，我觉得它最大的好处就是{angle}这一块做得顺。\n甲：不过我也不会乱花，账单我每个月都会看一下。\n乙：嗯，工具本身方便，关键还是看自己怎么用。",
        f"甲：你平时用{app}多吗？\n乙：挺多的，尤其是{scene}，手机一点就好了。\n甲：我之前还担心{angle}，后来发现只要按时看账单就还好。\n乙：对，正常消费就是图方便，不是说有额度就一定要用完。",
        f"甲：我觉得{app}跟以前现金支付比起来变化挺大的。\n乙：是啊，{scene}的时候不用找零，也能查到每一笔。\n甲：就是有时候看到{angle}这些功能，会忍不住研究一下。\n乙：研究可以，别超出自己承受范围就行。",
    ]
    return {"scenario": "normal_payment_app_discussion", "risk_terms": "支付宝;微信支付;花呗;借呗;额度;付款", "text": pick(templates, i)}


def normal_account_security_discussion(i: int) -> Dict[str, str]:
    topics = ["账号安全", "支付密码", "防火墙", "盗号新闻", "手机丢失", "登录提醒"]
    platforms = ["支付宝", "微信", "银行App", "校园卡系统", "公司财务系统", "购物平台"]
    opinions = ["现在平台做得比以前严", "双重验证有时候麻烦但有必要", "我一般会把通知打开", "只要不是自己操作就先查一下"]
    topic = pick(topics, i)
    platform = pick(platforms, i, 1)
    opinion = pick(opinions, i, 2)
    templates = [
        f"甲：上次听人说{platform}有{topic}的问题，我还专门去看了设置。\n乙：现在大家都比较在意这个。\n甲：我觉得{opinion}，至少自己心里有数。\n乙：对，平时聊这些更多是提醒自己注意。",
        f"甲：你说这些{platform}的{topic}，是不是年轻人反而更懂一点？\n乙：不一定，有些人天天用也不看设置。\n甲：我一般看到登录提醒会确认一下，不会随便改密码。\n乙：这就是正常使用习惯，谈不上什么大事。",
        f"甲：我刚才看到一个关于{topic}的帖子，里面提到{platform}。\n乙：这种帖子很多，有些讲得夸张。\n甲：嗯，我主要就是了解一下，毕竟日常付款都离不开它。\n乙：了解规则挺好，别被标题吓到就行。",
    ]
    return {"scenario": "normal_account_security_discussion", "risk_terms": "账号;安全;防火墙;盗号;支付密码", "text": pick(templates, i)}


def normal_credit_repayment_discussion(i: int) -> Dict[str, str]:
    products = ["花呗", "借呗", "信用卡", "车贷", "房贷", "助学贷款"]
    topics = ["分期", "提前还款", "利息", "还款日", "额度", "征信"]
    attitudes = ["能不用就少用", "急用时周转一下还可以", "每个月要算清楚", "不能这个月花下个月的钱"]
    product = pick(products, i)
    topic = pick(topics, i, 1)
    attitude = pick(attitudes, i, 2)
    templates = [
        f"甲：我最近在看{product}的{topic}规则，发现细节还挺多。\n乙：这些东西最好看清楚，不然容易超预算。\n甲：对，我现在的想法是{attitude}。\n乙：正常消费可以，关键是别形成恶性循环。",
        f"甲：你觉得{product}这种工具到底好不好？\n乙：看怎么用吧，{topic}这些规则要搞明白。\n甲：我身边有人用得很随意，我就不太敢。\n乙：谨慎点没错，毕竟最后都是要自己还。",
        f"甲：我以前觉得{product}只是方便，现在才知道{topic}也会影响判断。\n乙：对，尤其是金额积累起来以后。\n甲：所以我一般只拿它当临时工具。\n乙：这样比较稳。",
    ]
    return {"scenario": "normal_credit_repayment_discussion", "risk_terms": "花呗;借呗;信用卡;贷款;还款;征信;额度", "text": pick(templates, i)}


def normal_contract_salary_discussion(i: int) -> Dict[str, str]:
    entities = ["骑手工资", "合同金额", "项目提成", "社保", "五险一金", "外包费用"]
    amounts = ["三千左右", "六百一项", "一单多一块", "百分之八", "两千二底薪", "年底统一结算"]
    contexts = ["招聘方案", "合作合同", "工资结构", "项目预算", "福利政策", "报销流程"]
    entity = pick(entities, i)
    amount = pick(amounts, i, 1)
    context = pick(contexts, i, 2)
    templates = [
        f"甲：我们刚才讨论{context}，里面的{entity}要不要写细一点？\n乙：我觉得要写，像{amount}这种数字最好有依据。\n甲：对，不然后面审核的时候容易来回改。\n乙：合同和工资这些事情就是要说清楚。",
        f"甲：你看这个{context}，{entity}是不是偏低？\n乙：如果按市场行情，{amount}可能只是基础水平。\n甲：那我们要不要把社保和补贴也算进去？\n乙：可以，这样员工看起来更直观。",
        f"甲：今天会里一直在算{entity}，我听得有点乱。\n乙：其实就是把{context}拆开算，底薪、提成、社保分开。\n甲：嗯，金额不大但是项目多。\n乙：这种正常财务核算就很琐碎。",
    ]
    return {"scenario": "normal_contract_salary_discussion", "risk_terms": "合同;金额;工资;提成;社保;审核", "text": pick(templates, i)}


def normal_life_expense_discussion(i: int) -> Dict[str, str]:
    topics = ["彩礼", "婚礼预算", "房租", "装修", "学费", "家庭开销"]
    amounts = ["四十万", "二十万", "每月五千", "十几万", "六百块", "三千多"]
    tones = ["压力确实不小", "普通家庭很难一下拿出来", "还是要量力而行", "双方最好提前商量"]
    topic = pick(topics, i)
    amount = pick(amounts, i, 1)
    tone = pick(tones, i, 2)
    templates = [
        f"甲：我昨天看到一个帖子，说{topic}要{amount}，评论区吵得很厉害。\n乙：这种事情很容易引发争议。\n甲：我觉得{tone}，不能只看一个数字。\n乙：对，生活里的钱不是单纯越多越有面子。",
        f"甲：现在聊到{topic}，大家第一反应就是金额。\n乙：是啊，像{amount}这种数字听起来就很重。\n甲：但具体还要看家庭情况和城市。\n乙：没错，正常聊天也只能说说自己的看法。",
        f"甲：你觉得{topic}花到{amount}合理吗？\n乙：不好一概而论，有些地方习俗就是这样。\n甲：我还是觉得{tone}。\n乙：嗯，钱的事情最好别勉强。",
    ]
    return {"scenario": "normal_life_expense_discussion", "risk_terms": "彩礼;金额;家庭;开销;付款", "text": pick(templates, i)}


def normal_ecommerce_refund_discussion(i: int) -> Dict[str, str]:
    platforms = ["淘宝", "京东", "拼多多", "美团", "携程", "抖音商城"]
    topics = ["退款", "运费险", "售后", "发票", "订单取消", "理赔"]
    details = ["原路退回", "审核两天", "客服排队慢", "金额没对上", "系统显示处理中", "发票抬头写错"]
    platform = pick(platforms, i)
    topic = pick(topics, i, 1)
    detail = pick(details, i, 2)
    templates = [
        f"甲：我在{platform}上办{topic}，页面一直显示{detail}。\n乙：这种售后有时候就是慢。\n甲：我准备晚上再看一下，不行就找平台客服。\n乙：嗯，订单里的流程走完就行。",
        f"甲：你上次{platform}的{topic}多久到账？\n乙：好像两三天吧，主要看支付渠道。\n甲：我这个写着{detail}，所以有点不确定。\n乙：正常售后经常这样，等等看。",
        f"甲：这个{platform}{topic}我研究半天，发现规则挺复杂。\n乙：特别是{detail}这种情况，页面提示不太清楚。\n甲：对，不过最后应该还是按订单记录来。\n乙：是的。",
    ]
    return {"scenario": "normal_ecommerce_refund_discussion", "risk_terms": "退款;理赔;客服;订单;金额", "text": pick(templates, i)}


def normal_banking_admin_discussion(i: int) -> Dict[str, str]:
    services = ["银行卡限额", "公积金提取", "社保缴费", "个税申报", "医保报销", "跨行转账"]
    channels = ["手机银行", "政务App", "银行柜台", "公司系统", "社区窗口", "官方网站"]
    feelings = ["流程有点复杂", "材料要准备齐", "审核时间比较长", "页面经常看不懂", "每一步都要确认", "办完才放心"]
    service = pick(services, i)
    channel = pick(channels, i, 1)
    feeling = pick(feelings, i, 2)
    templates = [
        f"甲：我今天弄{service}，在{channel}上看了半天。\n乙：这些业务就是麻烦。\n甲：主要是{feeling}，不是特别熟的人很容易卡住。\n乙：正常，慢慢按页面做就行。",
        f"甲：你知道{service}要不要去{channel}吗？\n乙：看地区吧，有的线上能办，有的还要线下。\n甲：我看说明写得挺细，但还是怕漏材料。\n乙：这种行政类业务多核对几遍就好。",
        f"甲：我爸让我帮他看{service}，他说{channel}不会用。\n乙：年纪大一点确实不习惯。\n甲：我就帮他把材料列表记下来，周末再陪他弄。\n乙：可以，别着急。",
    ]
    return {"scenario": "normal_banking_admin_discussion", "risk_terms": "银行卡;限额;社保;公积金;转账;审核", "text": pick(templates, i)}


def normal_political_economy_discussion(i: int) -> Dict[str, str]:
    topics = ["大资本", "议会预算", "地方债", "能源价格", "金融政策", "国际新闻"]
    places = ["俄罗斯", "欧洲", "美国", "本地政府", "企业", "市场"]
    words = ["清除障碍", "资金压力", "经济损失", "风险控制", "安全保障", "利益分配"]
    topic = pick(topics, i)
    place = pick(places, i, 1)
    word = pick(words, i, 2)
    templates = [
        f"甲：昨天那个关于{place}的节目你看了吗？里面一直讲{topic}。\n乙：看了一点，感觉背景挺复杂。\n甲：主持人还提到{word}，但更多是在分析历史原因。\n乙：这种话题就是聊天了解一下。",
        f"甲：我最近对{topic}有点兴趣，尤其是{place}那边的情况。\n乙：你怎么突然看这个？\n甲：刷到一个访谈，里面说到{word}，听起来挺有意思。\n乙：当知识听听还行。",
        f"甲：他们讨论{place}的时候，总喜欢把{topic}和普通生活联系起来。\n乙：确实，最后都绕到钱和政策。\n甲：不过我只是听个大概，{word}这些词挺抽象的。\n乙：嗯，不是专业领域也不用太较真。",
    ]
    return {"scenario": "normal_political_economy_discussion", "risk_terms": "资本;资金;经济损失;风险控制;安全保障", "text": pick(templates, i)}


def fraud_payment_app_impersonation(i: int) -> Dict[str, str]:
    apps = ["支付宝", "微信支付", "银行App", "花呗", "借呗", "云闪付"]
    risks = ["账户异常", "额度冻结", "支付风控", "安全核验", "借贷记录异常", "交易拦截"]
    actions = ["把短信验证码读给我", "按我说的步骤转一笔验证金", "打开共享屏幕我远程看一下", "点开链接更新资料", "把银行卡号和余额确认一下", "下载我们的安全插件"]
    app = pick(apps, i)
    risk = pick(risks, i, 1)
    action = pick(actions, i, 2)
    templates = [
        f"甲：您好，这里是{app}安全中心，您名下出现{risk}，现在不处理会影响使用。\n乙：我刚才还正常付款啊。\n甲：系统刚触发的，您不要挂断，马上{action}，处理完就恢复。",
        f"甲：我们检测到您的{app}存在{risk}，需要做实时核验。\n乙：我自己去App里看看可以吗？\n甲：不行，页面已经被限制了，您现在必须跟着我{action}。",
        f"甲：您这个{app}风险记录今天必须清掉。\n乙：为什么这么急？\n甲：超过十分钟会冻结，您先{action}，后面我再解释。",
    ]
    return {"scenario": "fraud_payment_app_impersonation", "risk_terms": "支付宝;微信支付;花呗;借呗;账户异常;验证码;转账", "text": pick(templates, i)}


def fraud_refund_compensation(i: int) -> Dict[str, str]:
    platforms = ["淘宝", "京东", "美团", "携程", "快递公司", "保险平台"]
    reasons = ["订单退款失败", "理赔通道异常", "运费险赔付失败", "发票审核异常", "会员扣费错误", "航班退费异常"]
    actions = ["先转一笔保证金激活通道", "把验证码发给我确认身份", "下载会议软件共享屏幕", "点开我发的链接填写银行卡", "按提示把余额转到安全账户", "马上完成支付验证"]
    platform = pick(platforms, i)
    reason = pick(reasons, i, 1)
    action = pick(actions, i, 3)
    templates = [
        f"甲：您好，您在{platform}的{reason}，现在可以给您双倍补偿。\n乙：我没看到平台消息。\n甲：这是人工绿色通道，名额马上关闭，您现在{action}。",
        f"甲：这边是{platform}售后，您的{reason}卡住了。\n乙：那我在订单里申请就行吧？\n甲：订单里处理不了，必须电话里完成，您先{action}。",
        f"甲：我们要把{platform}的退款退给您，但是系统要求验证。\n乙：怎么验证？\n甲：很简单，您按照我说的{action}，不然钱退不回去。",
    ]
    return {"scenario": "fraud_refund_compensation", "risk_terms": "退款;理赔;保证金;验证码;链接;银行卡", "text": pick(templates, i)}


def fraud_loan_credit_pressure(i: int) -> Dict[str, str]:
    products = ["花呗", "借呗", "信用卡", "网贷额度", "征信记录", "消费贷"]
    threats = ["逾期记录上报", "额度永久关闭", "征信马上受影响", "账户会被冻结", "家属会收到通知", "贷款资格取消"]
    actions = ["转到指定账户做流水", "把验证码告诉我完成核验", "共享屏幕让我帮您关闭", "先交保证金恢复额度", "点击链接签署授权", "按我说的输入支付密码"]
    product = pick(products, i)
    threat = pick(threats, i, 1)
    action = pick(actions, i, 2)
    templates = [
        f"甲：您名下{product}出现异常，今天不处理就会{threat}。\n乙：我没有申请过啊。\n甲：所以才需要排查，您马上{action}，不然系统自动提交。",
        f"甲：我们是信贷处理中心，您的{product}风险等级升高。\n乙：我去官方客服问一下。\n甲：来不及了，窗口只有五分钟，先{action}。",
        f"甲：您这个{product}要做保护性关闭。\n乙：关闭为什么要操作钱？\n甲：这是系统流程，必须{action}，处理完原路退回。",
    ]
    return {"scenario": "fraud_loan_credit_pressure", "risk_terms": "花呗;借呗;贷款;征信;保证金;验证码;支付密码", "text": pick(templates, i)}


def fraud_contract_business_payment(i: int) -> Dict[str, str]:
    contexts = ["合同审核", "项目保证金", "政府补贴", "外包结算", "采购尾款", "频道租用合同"]
    identities = ["财务专员", "审核老师", "政府窗口", "项目负责人", "平台客服", "银行对公专员"]
    actions = ["先打款到临时监管账户", "把企业网银验证码发给我", "远程协助您提交付款", "马上扫码补缴手续费", "点链接确认收款账户", "按我说的流程转一笔测试款"]
    context = pick(contexts, i)
    identity = pick(identities, i, 1)
    action = pick(actions, i, 2)
    templates = [
        f"甲：我是{identity}，您这边的{context}卡在最后一步。\n乙：我们不是已经提交了吗？\n甲：系统显示金额校验失败，您现在{action}，否则今天过不了审。",
        f"甲：关于{context}，审核部门要求补一个资金证明。\n乙：我问下同事。\n甲：不要耽误，名额今天截止，您先{action}。",
        f"甲：您这个{context}涉及对公风险，需要电话里处理。\n乙：能不能走正式邮件？\n甲：邮件太慢了，您按我说的{action}，我这边马上放行。",
    ]
    return {"scenario": "fraud_contract_business_payment", "risk_terms": "合同;审核;金额;账户;转账;验证码;手续费", "text": pick(templates, i)}


def fraud_family_expense_emergency(i: int) -> Dict[str, str]:
    roles = ["孩子", "同学", "亲戚", "老师", "朋友", "同事"]
    emergencies = ["手机坏了", "交通事故", "学费补缴", "彩礼临时加钱", "住院押金", "账号被冻结"]
    actions = ["先转两万到这个卡号", "不要打电话确认", "马上扫码付款", "把验证码告诉我", "先借钱周转一下", "十分钟内处理"]
    role = pick(roles, i)
    emergency = pick(emergencies, i, 1)
    action = pick(actions, i, 2)
    templates = [
        f"甲：妈，我是你{role}，我这边{emergency}，现在很急。\n乙：你声音怎么不太一样？\n甲：我用别人手机打的，别问了，{action}，晚点我再解释。",
        f"甲：您好，我是您家里人托我联系的，他现在{emergency}。\n乙：我先确认一下。\n甲：他不方便接电话，您现在{action}，否则事情会变严重。",
        f"甲：这边学校通知{emergency}，名额马上截止。\n乙：我没收到正式通知。\n甲：系统可能延迟了，您先{action}，之后会有收据。",
    ]
    return {"scenario": "fraud_family_expense_emergency", "risk_terms": "彩礼;学费;押金;卡号;扫码;转账;验证码", "text": pick(templates, i)}


def fraud_account_security_remote(i: int) -> Dict[str, str]:
    platforms = ["银行账户", "支付宝账户", "微信账户", "社保账户", "公积金账户", "购物账户"]
    risks = ["异地登录", "资金风险", "异常交易", "资料被冒用", "安全锁定", "黑客攻击"]
    actions = ["共享屏幕让我看操作界面", "把验证码读出来", "把资金转到安全账户", "下载远程协助软件", "点击链接重新认证", "按我说的步骤关闭拦截"]
    platform = pick(platforms, i)
    risk = pick(risks, i, 1)
    action = pick(actions, i, 3)
    templates = [
        f"甲：您好，您的{platform}出现{risk}，需要立即排查。\n乙：我自己登录看看。\n甲：不能自己操作，容易触发冻结，您现在{action}。",
        f"甲：系统检测到{platform}被人尝试控制。\n乙：那我去网点行吗？\n甲：来不及，已经进入倒计时，您先{action}。",
        f"甲：我们是安全处理组，正在帮您保护{platform}。\n乙：我需要做什么？\n甲：不要挂断，马上{action}，否则资金损失由您自己承担。",
    ]
    return {"scenario": "fraud_account_security_remote", "risk_terms": "账户;异常交易;黑客;验证码;共享屏幕;安全账户", "text": pick(templates, i)}


NORMAL_SCENARIOS: List[Callable[[int], Dict[str, str]]] = [
    normal_payment_app_discussion,
    normal_account_security_discussion,
    normal_credit_repayment_discussion,
    normal_contract_salary_discussion,
    normal_life_expense_discussion,
    normal_ecommerce_refund_discussion,
    normal_banking_admin_discussion,
    normal_political_economy_discussion,
]

FRAUD_SCENARIOS: List[Callable[[int], Dict[str, str]]] = [
    fraud_payment_app_impersonation,
    fraud_refund_compensation,
    fraud_loan_credit_pressure,
    fraud_contract_business_payment,
    fraud_family_expense_emergency,
    fraud_account_security_remote,
]


def build_records() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    seen = set()
    next_normal_id = 1
    next_fraud_id = 1

    for scenario_fn in NORMAL_SCENARIOS:
        for local_idx in range(PER_NORMAL_SCENARIO):
            item = scenario_fn(local_idx)
            text = item["text"] + normal_memory_suffix(local_idx)
            norm = normalize_text(text)
            if norm in seen:
                raise ValueError(f"Duplicate normalized text in scenario {item['scenario']}")
            seen.add(norm)
            records.append(
                {
                    "sample_id": f"NFHN_{next_normal_id:04d}",
                    "case_type": "normal_finance_hard_negative",
                    "text_label": 0,
                    "split": split_for(local_idx),
                    "source": "semantic_contrast_normal_finance",
                    "source_metadata": "generated_by_evaluation/generate_benign_finance_hard_negatives.py",
                    "scenario": item["scenario"],
                    "risk_terms": item["risk_terms"],
                    "input_view": "full_dialogue_text",
                    "notes": "Training/dev only; realistic oral normal-finance discussion without fraud action chain.",
                    "text": text,
                }
            )
            next_normal_id += 1

    for scenario_fn in FRAUD_SCENARIOS:
        for local_idx in range(PER_FRAUD_SCENARIO):
            item = scenario_fn(local_idx)
            text = item["text"] + fraud_pressure_suffix(local_idx)
            norm = normalize_text(text)
            if norm in seen:
                raise ValueError(f"Duplicate normalized text in scenario {item['scenario']}")
            seen.add(norm)
            records.append(
                {
                    "sample_id": f"FCFP_{next_fraud_id:04d}",
                    "case_type": "finance_semantic_contrast_fraud",
                    "text_label": 1,
                    "split": split_for(local_idx),
                    "source": "semantic_contrast_fraud",
                    "source_metadata": "generated_by_evaluation/generate_benign_finance_hard_negatives.py",
                    "scenario": item["scenario"],
                    "risk_terms": item["risk_terms"],
                    "input_view": "full_dialogue_text",
                    "notes": "Training/dev only; same finance vocabulary with inducement, pressure, verification, transfer, or remote-control intent.",
                    "text": text,
                }
            )
            next_fraud_id += 1

    return records


def write_jsonl(records: Iterable[Dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(records: List[Dict[str, object]], path: Path) -> None:
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary(records: List[Dict[str, object]], path: Path) -> None:
    rows = []
    counters = {
        "split": Counter(str(r["split"]) for r in records),
        "label": Counter(str(r["text_label"]) for r in records),
        "case_type": Counter(str(r["case_type"]) for r in records),
        "source": Counter(str(r["source"]) for r in records),
        "scenario": Counter(str(r["scenario"]) for r in records),
    }
    rows.append({"scope": "overall", "key": "samples", "value": len(records)})
    for scope, counter in counters.items():
        for key, count in sorted(counter.items()):
            rows.append({"scope": scope, "key": key, "value": count})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "key", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    records = build_records()
    normal_records = [record for record in records if int(record["text_label"]) == 0]

    CONTRAST_JSONL.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, CONTRAST_JSONL)
    write_csv(records, CONTRAST_CSV)
    write_summary(records, CONTRAST_SUMMARY_CSV)

    write_jsonl(normal_records, LEGACY_BENIGN_JSONL)
    write_csv(normal_records, LEGACY_BENIGN_CSV)
    write_summary(normal_records, LEGACY_BENIGN_SUMMARY_CSV)

    print(f"Wrote semantic contrast records: {len(records)}")
    print(f"Contrast JSONL: {CONTRAST_JSONL}")
    print(f"Contrast CSV: {CONTRAST_CSV}")
    print(f"Contrast summary: {CONTRAST_SUMMARY_CSV}")
    print(f"Wrote benign-only compatibility records: {len(normal_records)}")
    print(f"Benign JSONL: {LEGACY_BENIGN_JSONL}")
    print("Split:", dict(Counter(str(r["split"]) for r in records)))
    print("Labels:", dict(Counter(int(r["text_label"]) for r in records)))
    print("Sources:", dict(Counter(str(r["source"]) for r in records)))


if __name__ == "__main__":
    main()
