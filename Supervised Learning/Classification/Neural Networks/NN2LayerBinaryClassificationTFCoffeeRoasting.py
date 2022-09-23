import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[185.31763812365773,12.693964573656494], [259.9204749802278,11.867663768540387], [231.0135710053633,14.414242107247178], [175.36664490449147,11.720586509332273], [187.12086466731515,14.12973205950237], [225.90586447535776,12.100249053221344], [208.40515675864629,14.17718919429808], [207.07593089146724,14.032737597096466], [280.6038535866463,14.2322592892385], [202.86935246580353,12.249010278880393], [196.70468985218398,13.54426389427831], [270.3132702825782,14.602255769890757], [192.94979108307552,15.196867586027297], [213.57283453112726,14.27503536880892], [164.4729866412344,11.918174233731765], [177.2575054164758,15.037798694753175], [241.7745473004188,14.896945294457739], [236.99889634274487,13.126169590702146], [219.73805620687455,13.873774071023625], [266.3859279625298,13.2527446646365], [270.4524148454968,13.95486775361035], [261.9630769838885,13.49222421966899], [243.4899477970464,12.85610149859724], [220.5818480297081,12.364893564076057], [163.59498627249437,11.654416519683876], [244.76317931210858,13.325722483526224], [271.1941098575787,14.84073281876912], [201.98784315491025,15.39471507882089], [229.92837149557923,14.563533256861191], [204.97123839474693,12.284679647873087], [173.18989703604765,12.224824904923947], [231.5137448304292,11.950531420445273], [152.6879510930519,14.831987856753681], [163.4205009209065,13.302338142207049], [215.94730736633585,13.98108963353425], [218.04195512421555,15.249372574915872], [251.3035402390422,13.797865991531863], [233.33173845521324,13.52620596642792], [280.24284419817286,12.406504253436546], [243.01866351505674,13.720386715591461], [155.67159152427263,12.684599999134935], [275.1675362792148,14.63825942972], [151.7321976266024,12.68650531865098], [151.32372212128612,14.809867771912312], [164.8996249617235,11.729820715099278], [282.5542513313383,13.283479517660854], [192.98305487359625,11.696053558252368], [202.59536338034098,12.964156229048553], [220.66990638748766,11.527141865925932], [169.9749888391795,12.339546102993625], [209.46811003107416,12.709213465239465], [232.79923632620387,12.641730422157181], [272.8004563613967,15.347479832792306], [158.02370683384683,12.336238876619916], [226.00812973954754,14.582640917935919], [158.64327122879146,12.23921073771952], [211.65721642824303,14.174763034200694], [271.94927014348957,14.9678918541662], [257.1569810814648,11.710928573506205], [281.8459265944453,13.958511796604627], [161.62563505219575,12.519715101559061], [233.80180141644928,13.042560467534559], [210.2914691912232,14.718340262395], [261.2441819509951,13.68714792072123], [256.98089904735,13.124019716372217], [281.5550480379854,13.920636661704144], [280.6392276627205,11.675950768225588], [269.16350813376454,13.737508750179506], [246.34126917943763,12.27037990151608], [224.07333575878408,12.657111696487677], [164.23986437700543,11.512747082472469], [272.42340268172154,14.18361726014331], [177.6779337749785,12.531274710250976], [212.85523205543603,14.773149013077894], [165.87945638859475,15.371139320654919], [277.4279534676567,12.478640382770534], [236.50555103276065,12.940390131064344], [244.13865150825384,11.847664400591015], [213.44539382786763,13.853963214760375], [234.57422434983218,14.271181896686253], [270.33648535513584,12.465000839629704], [170.68123283998017,13.062427748851903], [226.79179993803092,15.342115043682218], [245.91825405821646,14.453770202678516], [281.31680226443416,12.57097265103436], [185.02731944599353,13.190163350275608], [189.88170100166397,14.104413540913876], [278.48137931063735,12.114045970144097], [219.92293709971818,14.210312395192949], [216.57898498845285,15.15497536357607], [249.48122813719687,15.028707182442275], [165.08827339607947,12.283054013986838], [158.8700704564876,14.817277993858738], [279.9805193437852,11.555964084113734], [256.54924192266276,14.411324794111898], [272.6052172420259,12.581541893037603], [246.49491974773906,12.44969152587688], [160.2644820060435,14.480816250813096], [155.69875113941342,14.298377852630496], [188.26743273481935,13.44969358033831], [270.3569757726322,12.473631516000852], [213.22379155065659,12.920197788160978], [175.7014197283934,13.3946058733705], [174.52009414654242,14.69602996663428], [233.00092161976346,12.632523011343407], [281.3691743606422,12.881107377874972], [240.6196492618556,14.432894905937017], [185.80556267384637,11.54705520635124], [270.5031433454736,15.325666054366515], [172.9807912614288,12.114420844495411], [208.41010162485875,13.890278272598966], [283.51265469325926,15.353984469423763], [283.3601333927059,12.482277660949709], [230.84923224189154,13.243476574854926], [181.23930992461385,11.761509482866865], [172.77833045706768,12.933806915144826], [161.88293361414617,12.102955965479822], [156.02795049793286,13.991625094768693], [216.51672478192378,12.474212012399576], [221.0570798355102,13.197787110623912], [238.98568520018682,15.230668878991835], [197.6944343739337,14.080606130699557], [179.55375965839255,15.259597596063937], [233.38848486508607,12.134989750401584], [184.70189322163776,12.13660544245415], [174.1830946534081,12.72719542667368], [261.1145098270051,13.32823520652449], [187.41794560696474,13.176303443070145], [186.09876106325586,14.434729664693174], [157.93546834639034,12.656914240589117], [193.63822190218158,12.226078074717979], [249.65103075660312,12.220989450434844], [190.56498131738326,11.725906248510729], [252.00406101766478,12.955452066489801], [238.5503330231537,12.368944408787026], [152.94302628637445,12.78967263403881], [255.17362012759682,14.849783121859982], [197.0933671229322,14.887763114166347], [156.79710499295106,13.5884581442939], [184.7520262047317,13.256310754802325], [179.92164591974054,15.074266487197118], [190.79357984301413,15.281167708142997], [164.7271741542562,13.219324024518682], [209.86506596128294,14.337739172089272], [196.5780064220703,13.469979854431436], [159.510623172614,12.744125834662189], [247.87288291184055,11.9236455676461], [212.44231568694744,12.446907818273253], [172.34040751845203,11.98526205043424], [259.8719001968305,14.246628078158885], [201.22657353944328,13.066563771293861], [248.34175919001873,13.915824598478073], [273.66206057956043,15.177654744928303], [215.0935848814562,14.136541027876753], [223.53014138129097,12.741142166812214], [211.22431235280325,14.384703456843148], [224.6120906119603,14.029625465786433], [215.75489519873088,15.311910571776721], [254.82229579531514,12.02314013487207], [259.9047835847929,15.170310305641612], [260.2488557726487,12.87243299110953], [199.66771450418645,12.472857481264143], [157.52013585298806,13.388245382462747], [264.81482409816533,14.575246344636167], [239.3968517064999,14.888762677733132], [238.98037311101996,12.393333660898925], [258.42593537030615,12.970080017838821], [270.15836598717624,12.80593012341884], [162.40676332878297,14.419592780232279], [164.53231153619822,14.980857716307591], [205.60967939761395,14.620484796170247], [157.09674149202135,13.675351030031221], [241.38069603947136,12.018020515066691], [232.13370588852757,12.072090461366098], [191.038532155903,12.961140822110634], [233.6440250032013,12.020475379749087], [174.95146369503308,14.625026345239178], [246.64321151268967,13.316822678403394], [188.07040705409207,14.268818568957299], [213.15899444832192,12.746078692371208], [268.08223647809893,12.307361271926027], [258.5781853468444,13.971271622677762], [237.20731697867757,14.228609805051228], [251.01659085350278,15.023788380269526], [274.27882001758127,12.52195618668974], [172.1246376477015,15.085496334129594], [177.51695425291348,12.387859751661974], [258.70969378519794,15.364442353914889], [264.0136216597015,13.566921566247302], [200.70604970349163,15.454206928801511], [249.36929914354067,14.01637408426408], [151.50238376037396,12.280767115616436], [151.8213889559931,15.128161659699117], [181.9228573367597,12.184085241979979], [228.64664272643643,12.312407425124329], [223.78183257172367,15.299166804176906], [266.6276732858951,12.480510135072599], [273.68398194872805,13.097561760768226], [220.61000616523552,12.799890704802397], [284.99434167453603,12.728293824607729]])
Y = np.array([[1.], [0.], [0.], [0.], [1.], [1.], [0.], [0.], [0.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [0.], [0.], [0.], [0.], [0.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [0.], [0.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.], [0.], [0.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [0.], [1.], [1.], [0.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.], [0.], [0.], [0.], [1.], [1.], [0.], [0.], [1.], [0.], [0.], [1.], [0.], [0.], [0.], [1.], [0.], [0.], [0.], [0.], [1.], [0.], [0.], [0.], [0.], [1.], [0.], [0.], [1.], [0.], [0.], [1.], [0.], [0.], [0.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [0.], [0.], [0.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.], [1.], [1.], [0.], [0.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.], [0.], [0.], [0.], [1.], [0.]])
print(Y.shape)

x1= []
x2 = []
colors = ['red','green']
for i in range(len(X)):
  x1.append(X[i][0])
  x2.append(X[i][1])
plt.scatter(x1,x2,c= Y,cmap=matplotlib.colors.ListedColormap(colors),marker ='x')
plt.title('Coffee Roasting')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()


# NORMALIZING THE DATA
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)

#Tile/copy our data to increase the training set size and reduce the number of training epochs.
Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1))   
print(Xt.shape, Yt.shape)  

# TF Model
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
     ]
)
#Note 1: The tf.keras.Input(shape=(2,)), specifies the expected shape of the input. This allows Tensorflow to size the weights and bias parameters at this point. This is useful when exploring Tensorflow models. This statement can be omitted in practice and Tensorflow will size the network parameters when the input data is specified in the model.fit statement.
#Note 2: Including the sigmoid activation in the final layer is not considered best practice. It would instead be accounted for in the loss which improves numerical stability. This will be described in more detail in a later lab.
print(model.summary())
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

#The model.compile statement defines a loss function and specifies a compile optimization.

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),)
#The model.fit statement runs gradient descent and fits the weights to the data.

# Epochs = 10, Default TF Batch Size = 32, Training Examples = 200,000, Number of Batches = 200,000/32 = 6250 per epoch
model.fit(Xt,Yt,epochs=10,)

# UPDATED WEIGHTS
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

## Testing
X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")