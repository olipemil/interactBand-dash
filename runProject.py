from bandstructure_dash import Widget

#optional input
characterSi = ['Si s','Si pz','Si py','Si px','Si s','Si pz','Si py','Si px',"Si s*","Si s*"]
kpathSi = [[0.5, 0.5, 0.5],[0.0, 0.0, 0.0], [0.5, 0, 0.5], [3/4,1/4,1/2], [3/4,3/8,3/8] ,[0.0, 0.0, 0.0]]
k_labelSi = [r'L',"\u0393",r'X',r'W',r'K',"\u0393"]

#required input
wanDirect = "silicon_interact"
wanTag = "wannier90"
numWanSi = 8
#initialize the widget
model = Widget(wanDirect,wanTag,numWanSi,characterSi,kpathSi,k_labelSi)

# make the dash app with a flask server
import flask
from dash import Dash
server = flask.Flask(__name__)
dash_app = Dash(server=server, routes_pathname_prefix="/InteractBandstructure/")
dash_app = model.plotWidget(app=dash_app)

if __name__ == '__main__':
    dash_app.run_server(debug=True)

# just plot the bandstructure
#new_bsplot = model.plotBS()
#new_bsplot.show()
