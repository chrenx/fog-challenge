"use strict";var markdownitSynapse=(()=>{var T=Object.defineProperty;var J=Object.getOwnPropertyDescriptor;var Q=Object.getOwnPropertyNames;var X=Object.prototype.hasOwnProperty;var Y=(e,l)=>{for(var a in l)T(e,a,{get:l[a],enumerable:!0})},ee=(e,l,a,c)=>{if(l&&typeof l=="object"||typeof l=="function")for(let s of Q(l))!X.call(e,s)&&s!==a&&T(e,s,{get:()=>l[s],enumerable:!(c=J(l,s))||c.enumerable});return e};var ne=e=>ee(T({},"__esModule",{value:!0}),e);var be={};Y(be,{default:()=>Re,footnotes:()=>we,init_markdown_it:()=>ye,preprocessMarkdown:()=>me,resetFootnotes:()=>_e});var re=/\\([ \\!"#$%&'()*+,.\/:;<=>?@[\]^_`{|}~-])/g,ie="reference?",se="bookmark?",te=new RegExp("^syn([0-9]+[.]?[0-9]*)+"),oe=new RegExp(/^\/\w/),le=new RegExp("^([\\da-z\\.-]+)\\.([a-z\\.]{2,6})([\\/w \\.-]*)*\\/?.*$"),ae=new RegExp("^doi:10[.]{1}[0-9]+[/]{1}[a-zA-Z0-9_.]+$"),D=new RegExp("^\\s*(width[=]{1})?\\s*(.*)[}]{1}\\s*$"),O=new RegExp('^\\s*(text[=]{1}["]{1})?\\s*(.*)["]{1}[}]{1}\\s*$'),ue=new RegExp("^\\s*[*-+>]{1}[^|]*$"),ce=new RegExp("^\\s*\\w+\\s*[.)]{1}[^|]*$"),pe=new RegExp("^\\s*[`]{3}s*([a-zA-Z_0-9-]*)s*$"),A,R,v,k,S,E;function fe(e,l){let c=e.split("?")[1].split("&"),s=null,p;for(let f=0;f<c.length;f++)p=c[f].split("="),l===p[0]&&(s=p[1]);return s}function Z(e){if(e>=8192&&e<=8202)return!0;switch(e){case 9:case 10:case 11:case 12:case 13:case 32:case 160:case 5760:case 8239:case 8287:case 12288:return!0}return!1}function de(e){if(e>=65&&e<=90||e>=48&&e<=57||e>=97&&e<=122)return!0;switch(e){case 64:case 46:case 45:case 95:return!0}return!1}function ge(e,l,a){return a=a||0,e.substr(a,l.length)===l}function xe(e){return e.length>2?oe.test(e.substr(0,2)):!1}var he=function(e,l){let a=!1,c,s,p,f,w,y="widgetContainer",_=e.posMax,d=e.pos;if(d+3>=_||l)return!1;if(e.src.charCodeAt(d)===64){if(d>0&&!Z(e.src.charCodeAt(e.pos-1)))return!1;for(;e.pos<_&&!Z(e.src.charCodeAt(e.pos))&&de(e.src.charCodeAt(e.pos));)e.pos++;return c=e.src.slice(d+1,e.pos),e.posMax=e.pos,e.pos=d,s=e.push("synapse_open","span",1),s.markup="@",s.attrs=[["data-widgetparams","badge?alias="+c],["data-widget-type","badge"],["class",y],["id","widget-"+R+A]],s=e.push("link_open","a",1),s.attrs=[["href","https://www.synapse.org/Portal/aliasredirector?alias="+c]],s.markup="autolink",s.info="auto",s=e.push("text","",0),s.content="@"+c,s=e.push("link_close","a",-1),s.markup="autolink",s.info="auto",s=e.push("text","",0),s=e.push("synapse_close","span",-1),e.pos=e.posMax,e.posMax=_,R=R+1,!0}if(e.src.charCodeAt(d)!==36||e.src.charCodeAt(d+1)!==123)return!1;for(e.pos=d+2;e.pos<_;){if(e.src.charCodeAt(e.pos)===125){a=!0;break}e.md.inline.skipToken(e)}return!a||d+2===e.pos?(e.pos=d,!1):(c=e.src.slice(d+2,e.pos),p=c.replace(re,"$1"),c.lastIndexOf(ie,0)===0?(p+="&footnoteId="+k,s=e.push("synapse_reference_open","span",1),s.attrs=[["id","wikiReference"+k]],s=e.push("synapse_reference_close","span",-1),f=decodeURIComponent(p),w=fe(f,"text"),w&&(S+="${bookmark?text=["+k+"]&bookmarkID=wikiReference"+k+"} "+w+`
`),k++,y="inlineWidgetContainer"):c.lastIndexOf(se,0)===0&&(s=e.push("synapse_footnote_target_open","span",1),s.attrs=[["id","wikiFootnote"+k]],s=e.push("synapse_footnote_target_close","span",-1),k++,y="inlineWidgetContainer"),e.posMax=e.pos,e.pos=d+2,s=e.push("synapse_open","span",1),s.markup="${",s.attrs=[["data-widgetparams",p],["class",y],["id","widget-"+R+A],["data-widget-type",p.substring(0,p.indexOf("?"))]],s=e.push("text","",0),s.content="<Synapse widget>",s=e.push("synapse_close","span",-1),s.markup="}",e.pos=e.posMax+1,e.posMax=_,R=R+1,!0)};function ke(e,l,a){R=0,v=0,k=1,A=l,S="",E="",a&&(E=a),e.inline.ruler.after("emphasis","synapse",he)}function we(){return S}function _e(){S="",k=1}function me(e){let l="",a=e.split(`
`),c=!1,s=!1,p=!1;for(let f=0;f<a.length;f++)pe.test(a[f])&&(p=!p),p||(s=ue.test(a[f])||ce.test(a[f]),c&&!s&&(l+=`
`),c=s),l+=a[f]+`
`;return l}function ye(e,l,a,c,s,p,f,w,y,_,d){function q(){let n=e.renderer.rules.link_open||function(o,r,t,g,i){return i.renderToken(o,r,t)};e.renderer.rules.link_open=function(o,r,t,g,i){let u=o[r].attrIndex("target"),x=o[r].attrIndex("href"),m=ge(o[r].attrs[x][1],"#!"),I=xe(o[r].attrs[x][1]);return u<0?(x<0||!(I||m))&&(o[r].attrPush(["target","_blank"]),o[r].attrPush(["ref","noopener noreferrer"])):(o[r].attrs[u][1]="_blank",o[r].attrPush(["ref","noopener noreferrer"])),n(o,r,t,g,i)}}function F(){let n=e.renderer.rules.table_open||function(o,r,t,g,i){return i.renderToken(o,r,t)};e.renderer.rules.table_open=function(o,r,t,g,i){let u=o[r].attrIndex("class");return u<0?o[r].attrPush(["class","markdowntable"]):o[r].attrs[u][1]+=" markdowntable",n(o,r,t,g,i)}}function N(){e.linkify.set({fuzzyLink:!1}),e.linkify.add("syn",{validate:function(n,o,r){let t=n.slice(o);return r.re.synapse||(r.re.synapse=new RegExp("^([0-9]{3,}[.]?[0-9]*(\\/wiki\\/[0-9]+)?)+(?!_)(?=$|"+r.re.src_ZPCc+")")),r.re.synapse.test(t)?t.match(r.re.synapse)[0].length:0},normalize:function(n){n.url=E+"/Synapse:"+n.url.replace(/[.]/,"/version/")}}),e.linkify.add("doi:10.",{validate:function(n,o,r){let t=n.slice(o);return r.re.doi||(r.re.doi=new RegExp("^[0-9]+[/]{1}[a-zA-Z0-9_.]+(?!_)(?=$|"+r.re.src_ZPCc+")")),r.re.doi.test(t)?t.match(r.re.doi)[0].length:0},normalize:function(n){n.url="https://doi.org/"+n.url}})}let U=function(n,o){let r,t,g,i,u,x,m,I,C="",P=n.pos,z=n.pos,h=n.posMax,$=e.helpers.parseLinkLabel,H=e.helpers.parseLinkDestination,B=e.helpers.parseLinkTitle,L=e.utils.isSpace,G=e.utils.normalizeReference;if(n.src.charCodeAt(n.pos)!==91)return!1;let W=n.pos+1,b=$(n,n.pos,!0);if(b<0)return!1;if(i=b+1,i<h&&n.src.charCodeAt(i)===40){for(i++;i<h&&(t=n.src.charCodeAt(i),!(!L(t)&&t!==10));i++);if(i>=h)return!1;if(P=i,u=H(n.src,i,n.posMax),u.ok){let M=u.str;te.test(M)?u.str=E+"/Synapse:"+M.replace(/[.]/,"/version/"):le.test(M)?u.str="http://"+M:ae.test(M)&&(u.str="http://dx.doi.org/"+M),C=n.md.normalizeLink(u.str),n.md.validateLink(C)?i=u.pos:C=""}for(P=i;i<h&&(t=n.src.charCodeAt(i),!(!L(t)&&t!==10));i++);if(u=B(n.src,i,n.posMax),i<h&&P!==i&&u.ok)for(m=u.str,i=u.pos;i<h&&(t=n.src.charCodeAt(i),!(!L(t)&&t!==10));i++);else m="";if(i>=h||n.src.charCodeAt(i)!==41)return n.pos=z,!1;i++}else{if(typeof n.env.references>"u")return!1;if(i<h&&n.src.charCodeAt(i)===91?(P=i+1,i=$(n,i),i>=0?g=n.src.slice(P,i++):i=b+1):i=b+1,g||(g=n.src.slice(W,b)),x=n.env.references[G(g)],!x)return n.pos=z,!1;C=x.href,m=x.title}return o||(n.pos=W,n.posMax=b,I=n.push("link_open","a",1),I.attrs=r=[["href",C]],m&&r.push(["title",m]),n.md.inline.tokenize(n),I=n.push("link_close","a",-1)),n.pos=i,n.posMax=h,!0};function V(n,o,r){let t=r,g=o,i=n.charCodeAt(r);for(;t<g&&n.charCodeAt(t)===i;)t++;return{can_open:!0,can_close:!0,length:t-r}}let K=function(n,o){let r,t,g=n.pos,i=n.src.charCodeAt(g);if(o||i!==95&&i!==42)return!1;let u=V(n.src,n.posMax,n.pos);for(r=0;r<u.length;r++)t=n.push("text","",0),t.content=String.fromCharCode(i),n.delimiters.push({marker:i,jump:r,token:n.tokens.length-1,level:n.level,end:-1,open:u.can_open,close:u.can_close});return n.pos+=u.length,!0};function j(){e.set({html:!0,breaks:!0,linkify:!0}),e.disable(["heading"]),e.disable(["lheading"]),l&&e.use(l),a&&e.use(a),c&&e.use(c),s&&e.use(s),p&&e.use(p),f&&e.use(f),y&&e.use(y),_&&e.use(_),d&&e.use(d),w&&(e.use(w,"row",{marker:"{row}",minMarkerCount:1,render:function(n,o){let r;return n[o].nesting===1?r='<div class="container-fluid"><div class="row">':r=`</div></div>
`,r},validate:function(){return!0}}),e.use(w,"column",{marker:"{column",endMarker:"{column}",minMarkerCount:1,render:function(n,o){let r,t;return n[o].nesting===1?(r=D.exec(n[o].info),t='<div class="col-sm-'+e.utils.escapeHtml(r[2])+'">'):t=`</div>
`,t},validate:function(n){return D.test(n)}}),e.use(w,"nav",{marker:"{nav",endMarker:"{nav}",minMarkerCount:1,render:function(n,o){let r,t;return n[o].nesting===1?(r=O.exec(n[o].info),t='<div id="nav-target-'+v+A+'" target-text="'+e.utils.escapeHtml(r[2])+'">',v=v+1):t=`</div>
`,t},validate:function(n){return O.test(n)}})),q(),N(),F(),e.inline.ruler.at("link",U),e.inline.ruler.at("emphasis",K)}j()}var Re=ke;return ne(be);})();
markdownitSynapse = Object.assign(markdownitSynapse.default, markdownitSynapse)
