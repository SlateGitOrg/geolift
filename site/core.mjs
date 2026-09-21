
const n=(value,fallback=0)=>Number.isFinite(Number(value))?Number(value):fallback;
const clamp=(value,low,high)=>Math.min(high,Math.max(low,value));
const round=(value,digits=2)=>Number(value.toFixed(digits));
const mean=values=>values.length?values.reduce((sum,value)=>sum+value,0)/values.length:0;
const parseJSON=(value,fallback=[])=>{try{return JSON.parse(value)}catch{return fallback}};
const valuesFrom=value=>String(value).split(/[\s,]+/).map(Number).filter(Number.isFinite);
const result=(status,summary,metrics,rows,detail='')=>({status,summary,metrics,rows,detail});
const erf=x=>{const sign=x<0?-1:1,a=Math.abs(x),t=1/(1+0.3275911*a);const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-a*a);return sign*y};
const normalCdf=z=>0.5*(1+erf(z/Math.sqrt(2)));
const wilson=(successes,total)=>{if(!total)return[0,0];const z=1.96,p=successes/total,d=1+z*z/total,c=(p+z*z/(2*total))/d,h=z*Math.sqrt((p*(1-p)+z*z/(4*total))/total)/d;return[clamp(c-h,0,1),clamp(c+h,0,1)]};
const sha256=async value=>{const bytes=new TextEncoder().encode(String(value));const digest=await crypto.subtle.digest('SHA-256',bytes);return[...new Uint8Array(digest)].map(byte=>byte.toString(16).padStart(2,'0')).join('')};
const tag=(xml,name)=>xml.match(new RegExp('<'+name+'[^>]*>([\\s\\S]*?)<\\/'+name+'>','i'))?.[1]?.trim()??'';
const similarity=(a,b)=>{const x=String(a).toLowerCase(),y=String(b).toLowerCase();if(x===y)return 1;const A=new Set(x.split(/\W+/).filter(Boolean)),B=new Set(y.split(/\W+/).filter(Boolean));const inter=[...A].filter(v=>B.has(v)).length;return inter/Math.max(1,new Set([...A,...B]).size)};

export const meta={"slug":"geolift","name":"GeoLift","eyebrow":"Synthetic-control inference","description":"Gate a geo experiment on pre-period fit and calculate placebo-based incrementality.","fields":[{"name":"preRmse","label":"Pre-period RMSE","type":"number","min":0,"max":1,"step":0.001,"help":""},{"name":"fitGate","label":"Maximum accepted RMSE","type":"number","min":0,"max":1,"step":0.001,"help":""},{"name":"observedSales","label":"Observed post-period sales","type":"number","min":0,"max":1000000000,"step":1000,"help":""},{"name":"syntheticSales","label":"Synthetic-control sales","type":"number","min":0,"max":1000000000,"step":1000,"help":""},{"name":"placeboGaps","label":"Placebo gaps","type":"textarea","rows":4,"help":"Comma-separated absolute sales gaps"}]};
export const initialState={"preRmse":0.034,"fitGate":0.08,"observedSales":1045000,"syntheticSales":1000000,"placeboGaps":"12000, 18000, 25000, 31000, 9000, 15000, 22000, 28000, 11000"};
export const alternateState={"preRmse":0.14,"fitGate":0.08,"observedSales":1045000,"syntheticSales":1000000,"placeboGaps":"12000,18000,25000"};
export async function compute(i){const rmse=n(i.preRmse),gate=n(i.fitGate),obs=n(i.observedSales),syn=n(i.syntheticSales),gap=obs-syn,places=valuesFrom(i.placeboGaps).map(Math.abs),p=(1+places.filter(v=>v>=Math.abs(gap)).length)/(places.length+1),fit=rmse<=gate,status=!fit?'Reject estimate':p<=.1?'Publish lift':'Inconclusive';return result(status,fit?`Estimated incremental lift is ${round(gap/syn*100)}% with placebo p=${round(p,3)}.`:`Pre-period fit ${round(rmse,3)} exceeds the ${round(gate,3)} gate.`,[{label:'Incremental sales',value:Math.round(gap).toLocaleString()},{label:'Lift',value:`${round(gap/syn*100)}%`},{label:'Placebo p',value:round(p,3)},{label:'Fit ratio',value:round(rmse/gate,2)}],places.map((value,index)=>({placebo:index+1,gap:value,beatsTreatment:value>=Math.abs(gap)?'Yes':'No'})),fit?'Fit gate passed before inference.':'Inference suppressed until fit improves.')}
