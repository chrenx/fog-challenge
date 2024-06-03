!function(e) {
    if ("object" == typeof exports && "undefined" != typeof module) module.exports = e(); else if ("function" == typeof define && define.amd) define([], e); else {
        ("undefined" != typeof window ? window : "undefined" != typeof global ? global : "undefined" != typeof self ? self : this).markdownitMath = e();
    }
}(function() {
    return function e(n, r, t) {
        function o(s, f) {
            if (!r[s]) {
                if (!n[s]) {
                    var u = "function" == typeof require && require;
                    if (!f && u) return u(s, !0);
                    if (i) return i(s, !0);
                    var l = new Error("Cannot find module '" + s + "'");
                    throw l.code = "MODULE_NOT_FOUND", l;
                }
                var c = r[s] = {
                    exports: {}
                };
                n[s][0].call(c.exports, function(e) {
                    var r = n[s][1][e];
                    return o(r || e);
                }, c, c.exports, e, n, r, t);
            }
            return r[s].exports;
        }
        for (var i = "function" == typeof require && require, s = 0; s < t.length; s++) o(t[s]);
        return o;
    }({
        1: [ function(e, n, r) {
            "use strict";
            var t = "mathjax-", o = 0;
            function i(e, n) {
                var r, t = !1, o = e.posMax, i = e.pos, s = "$", f = "$";
                if (36 !== e.src.charCodeAt(i) || i >= o) return !1;
                if (e.pos + 1 < e.posMax) if (36 === e.src.charCodeAt(i + 1)) s = "$$", f = "$$"; else if (123 === e.src.charCodeAt(i + 1)) return !1;
                if (e.src.slice(e.pos, e.pos + s.length) !== s) return !1;
                if (n) return !1;
                if (i + s.length >= o) return !1;
                for (e.pos = i + s.length; e.pos < o; ) {
                    if (e.src.slice(e.pos, e.pos + f.length) === f) {
                        t = !0;
                        break;
                    }
                    e.md.inline.skipToken(e);
                }
                return t ? (r = e.src.slice(i + s.length, e.pos)).indexOf("\n") > -1 ? (e.pos = i, 
                !1) : (e.posMax = e.pos, e.pos = i + s.length, e.push("math_inline", "math", 0).content = r, 
                e.pos = e.posMax + f.length, e.posMax = o, !0) : (e.pos = i, !1);
            }
            function s(e, n, r, t) {
                var o, i, s = !1, f = e.bMarks[n] + e.tShift[n];
                if (r <= n + 3 || "$$\n" !== e.src.slice(f, f + "$$\n".length)) return !1;
                if (t) return !0;
                for (o = n; !s; ) {
                    if (++o > r) return !1;
                    if (f = e.bMarks[o] + e.tShift[o], o === r) {
                        if (e.src.slice(f - "$$\n".trim().length, f) !== "$$\n".trim()) return !1;
                        o--, s = !0;
                    } else if ("$$\n" !== e.src.slice(f, f + "$$\n".length)) continue;
                    f += "$$\n".length, f = e.skipSpaces(f), s = !0;
                }
                return e.line = o + 1, (i = e.push("math_block", "math", 0)).block = !0, i.content = e.getLines(n + 1, o, 0, !0), 
                i.info = void 0, i.markup = "$$\n", !0;
            }
            n.exports = function(e, n) {
                o = 0, e.inline.ruler.before("escape", "math_inline", i), e.block.ruler.after("blockquote", "math_block", s, {
                    alt: [ "paragraph", "reference", "blockquote", "list" ]
                });
                var r = function(e) {
                    return function(n, r) {
                        return '<span id="' + t + o++ + e + '" class="math">' + n[r].content + "</span>";
                    };
                }(n);
                e.renderer.rules.math_inline = r, e.renderer.rules.math_block = r;
            };
        }, {} ]
    }, {}, [ 1 ])(1);
});
