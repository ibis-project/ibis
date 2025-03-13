var _JUPYTERLAB;
(() => { // webpackBootstrap
"use strict";
var __webpack_modules__ = ({
"95": (function (__unused_webpack_module, exports, __webpack_require__) {

var moduleMap = {
  "./extension": function() {
return Promise.all([__webpack_require__.e("154"), __webpack_require__.e("814")]).then(function() { return function() { return __webpack_require__(148); }; });
},
};
var get = function(module, getScope) {
  __webpack_require__.R = getScope;
  getScope = (
    __webpack_require__.o(moduleMap, module)
      ? moduleMap[module]()
      : Promise.resolve().then(() => {
        throw new Error('Module "' + module + '" does not exist in container.');
      })
  );
  __webpack_require__.R = undefined;
  return getScope;
}
var init = function(shareScope, initScope) {
  if (!__webpack_require__.S) return;
  var name = "default";
  var oldScope = __webpack_require__.S[name];
  if(oldScope && oldScope !== shareScope) throw new Error("Container initialization failed as it has already been initialized with a different share scope");
  __webpack_require__.S[name] = shareScope;
  return __webpack_require__.I(name, initScope);
}
__webpack_require__.d(exports, {
	get: () => (get),
	init: () => (init)
});

}),

});
/************************************************************************/
// The module cache
var __webpack_module_cache__ = {};

// The require function
function __webpack_require__(moduleId) {

// Check if module is in cache
var cachedModule = __webpack_module_cache__[moduleId];
if (cachedModule !== undefined) {
return cachedModule.exports;
}
// Create a new module (and put it into the cache)
var module = (__webpack_module_cache__[moduleId] = {
exports: {}
});
// Execute the module function
__webpack_modules__[moduleId](module, module.exports, __webpack_require__);

// Return the exports of the module
return module.exports;

}

// expose the modules object (__webpack_modules__)
__webpack_require__.m = __webpack_modules__;

// expose the module cache
__webpack_require__.c = __webpack_module_cache__;

/************************************************************************/
// webpack/runtime/create_fake_namespace_object
(() => {
var getProto = Object.getPrototypeOf ? function(obj) { return Object.getPrototypeOf(obj); } : function(obj) { return obj.__proto__ };
var leafPrototypes;
// create a fake namespace object
// mode & 1: value is a module id, require it
// mode & 2: merge all properties of value into the ns
// mode & 4: return value when already ns object
// mode & 16: return value when it's Promise-like
// mode & 8|1: behave like require
__webpack_require__.t = function(value, mode) {
	if(mode & 1) value = this(value);
	if(mode & 8) return value;
	if(typeof value === 'object' && value) {
		if((mode & 4) && value.__esModule) return value;
		if((mode & 16) && typeof value.then === 'function') return value;
	}
	var ns = Object.create(null);
	__webpack_require__.r(ns);
	var def = {};
	leafPrototypes = leafPrototypes || [null, getProto({}), getProto([]), getProto(getProto)];
	for(var current = mode & 2 && value; typeof current == 'object' && !~leafPrototypes.indexOf(current); current = getProto(current)) {
		Object.getOwnPropertyNames(current).forEach(function(key) { def[key] = function() { return  value[key]; } });
	}
	def['default'] = function() { return value };
	__webpack_require__.d(ns, def);
	return ns;
};
})();
// webpack/runtime/define_property_getters
(() => {
__webpack_require__.d = function(exports, definition) {
	for(var key in definition) {
        if(__webpack_require__.o(definition, key) && !__webpack_require__.o(exports, key)) {
            Object.defineProperty(exports, key, { enumerable: true, get: definition[key] });
        }
    }
};
})();
// webpack/runtime/ensure_chunk
(() => {
__webpack_require__.f = {};
// This file contains only the entry chunk.
// The chunk loading function for additional chunks
__webpack_require__.e = function (chunkId) {
	return Promise.all(
		Object.keys(__webpack_require__.f).reduce(function (promises, key) {
			__webpack_require__.f[key](chunkId, promises);
			return promises;
		}, [])
	);
};

})();
// webpack/runtime/get css chunk filename
(() => {
// This function allow to reference chunks
        __webpack_require__.k = function (chunkId) {
          // return url for filenames not based on template
          
          // return url for filenames based on template
          return "" + chunkId + ".css";
        };
      
})();
// webpack/runtime/get javascript chunk filename
(() => {
// This function allow to reference chunks
        __webpack_require__.u = function (chunkId) {
          // return url for filenames not based on template
          
          // return url for filenames based on template
          return "" + chunkId + "." + {"154": "6fb5ba6e","814": "869fa0ef",}[chunkId] + ".js";
        };
      
})();
// webpack/runtime/global
(() => {
__webpack_require__.g = (function () {
	if (typeof globalThis === 'object') return globalThis;
	try {
		return this || new Function('return this')();
	} catch (e) {
		if (typeof window === 'object') return window;
	}
})();

})();
// webpack/runtime/has_own_property
(() => {
__webpack_require__.o = function (obj, prop) {
	return Object.prototype.hasOwnProperty.call(obj, prop);
};

})();
// webpack/runtime/load_script
(() => {
var inProgress = {};

var dataWebpackPrefix = "@anywidget/monorepo:";
// loadScript function to load a script via script tag
__webpack_require__.l = function (url, done, key, chunkId) {
	if (inProgress[url]) {
		inProgress[url].push(done);
		return;
	}
	var script, needAttach;
	if (key !== undefined) {
		var scripts = document.getElementsByTagName("script");
		for (var i = 0; i < scripts.length; i++) {
			var s = scripts[i];
			if (s.getAttribute("src") == url || s.getAttribute("data-webpack") == dataWebpackPrefix + key) {
				script = s;
				break;
			}
		}
	}
	if (!script) {
		needAttach = true;
		script = document.createElement('script');
		
		script.charset = 'utf-8';
		script.timeout = 120;
		if (__webpack_require__.nc) {
			script.setAttribute("nonce", __webpack_require__.nc);
		}
		script.setAttribute("data-webpack", dataWebpackPrefix + key);
		script.src = url;

		
	}
	inProgress[url] = [done];
	var onScriptComplete = function (prev, event) {
		script.onerror = script.onload = null;
		clearTimeout(timeout);
		var doneFns = inProgress[url];
		delete inProgress[url];
		script.parentNode && script.parentNode.removeChild(script);
		doneFns &&
			doneFns.forEach(function (fn) {
				return fn(event);
			});
		if (prev) return prev(event);
	};
	var timeout = setTimeout(
		onScriptComplete.bind(null, undefined, {
			type: 'timeout',
			target: script
		}),
		120000
	);
	script.onerror = onScriptComplete.bind(null, script.onerror);
	script.onload = onScriptComplete.bind(null, script.onload);
	needAttach && document.head.appendChild(script);
};

})();
// webpack/runtime/make_namespace_object
(() => {
// define __esModule on exports
__webpack_require__.r = function(exports) {
	if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
		Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
	}
	Object.defineProperty(exports, '__esModule', { value: true });
};

})();
// webpack/runtime/sharing
(() => {

__webpack_require__.S = {};
__webpack_require__.initializeSharingData = { scopeToSharingDataMapping: {  }, uniqueName: "@anywidget/monorepo" };
var initPromises = {};
var initTokens = {};
__webpack_require__.I = function(name, initScope) {
	if (!initScope) initScope = [];
	// handling circular init calls
	var initToken = initTokens[name];
	if (!initToken) initToken = initTokens[name] = {};
	if (initScope.indexOf(initToken) >= 0) return;
	initScope.push(initToken);
	// only runs once
	if (initPromises[name]) return initPromises[name];
	// creates a new share scope if needed
	if (!__webpack_require__.o(__webpack_require__.S, name))
		__webpack_require__.S[name] = {};
	// runs all init snippets from all modules reachable
	var scope = __webpack_require__.S[name];
	var warn = function (msg) {
		if (typeof console !== "undefined" && console.warn) console.warn(msg);
	};
	var uniqueName = __webpack_require__.initializeSharingData.uniqueName;
	var register = function (name, version, factory, eager) {
		var versions = (scope[name] = scope[name] || {});
		var activeVersion = versions[version];
		if (
			!activeVersion ||
			(!activeVersion.loaded &&
				(!eager != !activeVersion.eager
					? eager
					: uniqueName > activeVersion.from))
		)
			versions[version] = { get: factory, from: uniqueName, eager: !!eager };
	};
	var initExternal = function (id) {
		var handleError = function (err) {
			warn("Initialization of sharing external failed: " + err);
		};
		try {
			var module = __webpack_require__(id);
			if (!module) return;
			var initFn = function (module) {
				return (
					module &&
					module.init &&
					module.init(__webpack_require__.S[name], initScope)
				);
			};
			if (module.then) return promises.push(module.then(initFn, handleError));
			var initResult = initFn(module);
			if (initResult && initResult.then)
				return promises.push(initResult["catch"](handleError));
		} catch (err) {
			handleError(err);
		}
	};
	var promises = [];
	var scopeToSharingDataMapping = __webpack_require__.initializeSharingData.scopeToSharingDataMapping;
	if (scopeToSharingDataMapping[name]) {
		scopeToSharingDataMapping[name].forEach(function (stage) {
			if (typeof stage === "object") register(stage.name, stage.version, stage.factory, stage.eager);
			else initExternal(stage)
		});
	}
	if (!promises.length) return (initPromises[name] = 1);
	return (initPromises[name] = Promise.all(promises).then(function () {
		return (initPromises[name] = 1);
	}));
};


})();
// webpack/runtime/auto_public_path
(() => {

    var scriptUrl;
    if (__webpack_require__.g.importScripts) scriptUrl = __webpack_require__.g.location + "";
    var document = __webpack_require__.g.document;
    if (!scriptUrl && document) {
      if (document.currentScript) scriptUrl = document.currentScript.src;
        if (!scriptUrl) {
          var scripts = document.getElementsByTagName("script");
              if (scripts.length) {
                var i = scripts.length - 1;
                while (i > -1 && !scriptUrl) scriptUrl = scripts[i--].src;
              }
        }
      }
    
    // When supporting browsers where an automatic publicPath is not supported you must specify an output.publicPath manually via configuration",
    // or pass an empty string ("") and set the __webpack_public_path__ variable from your code to use your own logic.',
    if (!scriptUrl) throw new Error("Automatic publicPath is not supported in this browser");
    scriptUrl = scriptUrl.replace(/#.*$/, "").replace(/\?.*$/, "").replace(/\/[^\/]+$/, "/");
    __webpack_require__.p = scriptUrl
    
})();
// webpack/runtime/consumes_loading
(() => {

__webpack_require__.consumesLoadingData = { chunkMapping: {"814":["397"]}, moduleIdToConsumeDataMapping: { "397": { shareScope: "default", shareKey: "@jupyter-widgets/base", import: null, requiredVersion: "^6", strictVersion: false, singleton: true, eager: false, fallback: undefined } }, initialConsumes: [] };
var splitAndConvert = function(str) {
  return str.split(".").map(function(item) {
    return +item == item ? +item : item;
  });
};
var parseRange = function(str) {
  // see https://docs.npmjs.com/misc/semver#range-grammar for grammar
  var parsePartial = function(str) {
    var match = /^([^-+]+)?(?:-([^+]+))?(?:\+(.+))?$/.exec(str);
    var ver = match[1] ? [0].concat(splitAndConvert(match[1])) : [0];
    if (match[2]) {
      ver.length++;
      ver.push.apply(ver, splitAndConvert(match[2]));
    }

    // remove trailing any matchers
    let last = ver[ver.length - 1];
    while (
      ver.length &&
      (last === undefined || /^[*xX]$/.test(/** @type {string} */ (last)))
    ) {
      ver.pop();
      last = ver[ver.length - 1];
    }

    return ver;
  };
  var toFixed = function(range) {
    if (range.length === 1) {
      // Special case for "*" is "x.x.x" instead of "="
      return [0];
    } else if (range.length === 2) {
      // Special case for "1" is "1.x.x" instead of "=1"
      return [1].concat(range.slice(1));
    } else if (range.length === 3) {
      // Special case for "1.2" is "1.2.x" instead of "=1.2"
      return [2].concat(range.slice(1));
    } else {
      return [range.length].concat(range.slice(1));
    }
  };
  var negate = function(range) {
    return [-range[0] - 1].concat(range.slice(1));
  };
  var parseSimple = function(str) {
    // simple       ::= primitive | partial | tilde | caret
    // primitive    ::= ( '<' | '>' | '>=' | '<=' | '=' | '!' ) ( ' ' ) * partial
    // tilde        ::= '~' ( ' ' ) * partial
    // caret        ::= '^' ( ' ' ) * partial
    const match = /^(\^|~|<=|<|>=|>|=|v|!)/.exec(str);
    const start = match ? match[0] : "";
    const remainder = parsePartial(
      start.length ? str.slice(start.length).trim() : str.trim()
    );
    switch (start) {
      case "^":
        if (remainder.length > 1 && remainder[1] === 0) {
          if (remainder.length > 2 && remainder[2] === 0) {
            return [3].concat(remainder.slice(1));
          }
          return [2].concat(remainder.slice(1));
        }
        return [1].concat(remainder.slice(1));
      case "~":
        return [2].concat(remainder.slice(1));
      case ">=":
        return remainder;
      case "=":
      case "v":
      case "":
        return toFixed(remainder);
      case "<":
        return negate(remainder);
      case ">": {
        // and( >=, not( = ) ) => >=, =, not, and
        const fixed = toFixed(remainder);
        return [, fixed, 0, remainder, 2];
      }
      case "<=":
        // or( <, = ) => <, =, or
        return [, toFixed(remainder), negate(remainder), 1];
      case "!": {
        // not =
        const fixed = toFixed(remainder);
        return [, fixed, 0];
      }
      default:
        throw new Error("Unexpected start value");
    }
  };
  var combine = function(items, fn) {
    if (items.length === 1) return items[0];
    const arr = [];
    for (const item of items.slice().reverse()) {
      if (0 in item) {
        arr.push(item);
      } else {
        arr.push.apply(arr, item.slice(1));
      }
    }
    return [,].concat(arr, items.slice(1).map(() => fn));
  };
  var parseRange = function(str) {
    // range      ::= hyphen | simple ( ' ' ( ' ' ) * simple ) * | ''
    // hyphen     ::= partial ( ' ' ) * ' - ' ( ' ' ) * partial
    const items = str.split(/\s+-\s+/);
    if (items.length === 1) {
      const items = str
        .trim()
        .split(/(?<=[-0-9A-Za-z])\s+/g)
        .map(parseSimple);
      return combine(items, 2);
    }
    const a = parsePartial(items[0]);
    const b = parsePartial(items[1]);
    // >=a <=b => and( >=a, or( <b, =b ) ) => >=a, <b, =b, or, and
    return [, toFixed(b), negate(b), 1, a, 2];
  };
  var parseLogicalOr = function(str) {
    // range-set  ::= range ( logical-or range ) *
    // logical-or ::= ( ' ' ) * '||' ( ' ' ) *
    const items = str.split(/\s*\|\|\s*/).map(parseRange);
    return combine(items, 1);
  };
  return parseLogicalOr(str);
};
var parseVersion = function(str) {
	var match = /^([^-+]+)?(?:-([^+]+))?(?:\+(.+))?$/.exec(str);
	/** @type {(string|number|undefined|[])[]} */
	var ver = match[1] ? splitAndConvert(match[1]) : [];
	if (match[2]) {
		ver.length++;
		ver.push.apply(ver, splitAndConvert(match[2]));
	}
	if (match[3]) {
		ver.push([]);
		ver.push.apply(ver, splitAndConvert(match[3]));
	}
	return ver;
}
var versionLt = function(a, b) {
	a = parseVersion(a);
	b = parseVersion(b);
	var i = 0;
	for (;;) {
		// a       b  EOA     object  undefined  number  string
		// EOA        a == b  a < b   b < a      a < b   a < b
		// object     b < a   (0)     b < a      a < b   a < b
		// undefined  a < b   a < b   (0)        a < b   a < b
		// number     b < a   b < a   b < a      (1)     a < b
		// string     b < a   b < a   b < a      b < a   (1)
		// EOA end of array
		// (0) continue on
		// (1) compare them via "<"

		// Handles first row in table
		if (i >= a.length) return i < b.length && (typeof b[i])[0] != "u";

		var aValue = a[i];
		var aType = (typeof aValue)[0];

		// Handles first column in table
		if (i >= b.length) return aType == "u";

		var bValue = b[i];
		var bType = (typeof bValue)[0];

		if (aType == bType) {
			if (aType != "o" && aType != "u" && aValue != bValue) {
				return aValue < bValue;
			}
			i++;
		} else {
			// Handles remaining cases
			if (aType == "o" && bType == "n") return true;
			return bType == "s" || aType == "u";
		}
	}
}
var rangeToString = function(range) {
	var fixCount = range[0];
	var str = "";
	if (range.length === 1) {
		return "*";
	} else if (fixCount + 0.5) {
		str +=
			fixCount == 0
				? ">="
				: fixCount == -1
				? "<"
				: fixCount == 1
				? "^"
				: fixCount == 2
				? "~"
				: fixCount > 0
				? "="
				: "!=";
		var needDot = 1;
		for (var i = 1; i < range.length; i++) {
			var item = range[i];
			var t = (typeof item)[0];
			needDot--;
			str +=
				t == "u"
					? // undefined: prerelease marker, add an "-"
					  "-"
					: // number or string: add the item, set flag to add an "." between two of them
					  (needDot > 0 ? "." : "") + ((needDot = 2), item);
		}
		return str;
	} else {
		var stack = [];
		for (var i = 1; i < range.length; i++) {
			var item = range[i];
			stack.push(
				item === 0
					? "not(" + pop() + ")"
					: item === 1
					? "(" + pop() + " || " + pop() + ")"
					: item === 2
					? stack.pop() + " " + stack.pop()
					: rangeToString(item)
			);
		}
		return pop();
	}
	function pop() {
		return stack.pop().replace(/^\((.+)\)$/, "$1");
	}
}
var satisfy = function(range, version) {
	if (0 in range) {
		version = parseVersion(version);
		var fixCount = /** @type {number} */ (range[0]);
		// when negated is set it swill set for < instead of >=
		var negated = fixCount < 0;
		if (negated) fixCount = -fixCount - 1;
		for (var i = 0, j = 1, isEqual = true; ; j++, i++) {
			// cspell:word nequal nequ

			// when isEqual = true:
			// range         version: EOA/object  undefined  number    string
			// EOA                    equal       block      big-ver   big-ver
			// undefined              bigger      next       big-ver   big-ver
			// number                 smaller     block      cmp       big-cmp
			// fixed number           smaller     block      cmp-fix   differ
			// string                 smaller     block      differ    cmp
			// fixed string           smaller     block      small-cmp cmp-fix

			// when isEqual = false:
			// range         version: EOA/object  undefined  number    string
			// EOA                    nequal      block      next-ver  next-ver
			// undefined              nequal      block      next-ver  next-ver
			// number                 nequal      block      next      next
			// fixed number           nequal      block      next      next   (this never happens)
			// string                 nequal      block      next      next
			// fixed string           nequal      block      next      next   (this never happens)

			// EOA end of array
			// equal (version is equal range):
			//   when !negated: return true,
			//   when negated: return false
			// bigger (version is bigger as range):
			//   when fixed: return false,
			//   when !negated: return true,
			//   when negated: return false,
			// smaller (version is smaller as range):
			//   when !negated: return false,
			//   when negated: return true
			// nequal (version is not equal range (> resp <)): return true
			// block (version is in different prerelease area): return false
			// differ (version is different from fixed range (string vs. number)): return false
			// next: continues to the next items
			// next-ver: when fixed: return false, continues to the next item only for the version, sets isEqual=false
			// big-ver: when fixed || negated: return false, continues to the next item only for the version, sets isEqual=false
			// next-nequ: continues to the next items, sets isEqual=false
			// cmp (negated === false): version < range => return false, version > range => next-nequ, else => next
			// cmp (negated === true): version > range => return false, version < range => next-nequ, else => next
			// cmp-fix: version == range => next, else => return false
			// big-cmp: when negated => return false, else => next-nequ
			// small-cmp: when negated => next-nequ, else => return false

			var rangeType = j < range.length ? (typeof range[j])[0] : "";

			var versionValue;
			var versionType;

			// Handles first column in both tables (end of version or object)
			if (
				i >= version.length ||
				((versionValue = version[i]),
				(versionType = (typeof versionValue)[0]) == "o")
			) {
				// Handles nequal
				if (!isEqual) return true;
				// Handles bigger
				if (rangeType == "u") return j > fixCount && !negated;
				// Handles equal and smaller: (range === EOA) XOR negated
				return (rangeType == "") != negated; // equal + smaller
			}

			// Handles second column in both tables (version = undefined)
			if (versionType == "u") {
				if (!isEqual || rangeType != "u") {
					return false;
				}
			}

			// switch between first and second table
			else if (isEqual) {
				// Handle diagonal
				if (rangeType == versionType) {
					if (j <= fixCount) {
						// Handles "cmp-fix" cases
						if (versionValue != range[j]) {
							return false;
						}
					} else {
						// Handles "cmp" cases
						if (negated ? versionValue > range[j] : versionValue < range[j]) {
							return false;
						}
						if (versionValue != range[j]) isEqual = false;
					}
				}

				// Handle big-ver
				else if (rangeType != "s" && rangeType != "n") {
					if (negated || j <= fixCount) return false;
					isEqual = false;
					j--;
				}

				// Handle differ, big-cmp and small-cmp
				else if (j <= fixCount || versionType < rangeType != negated) {
					return false;
				} else {
					isEqual = false;
				}
			} else {
				// Handles all "next-ver" cases in the second table
				if (rangeType != "s" && rangeType != "n") {
					isEqual = false;
					j--;
				}

				// next is applied by default
			}
		}
	}
	/** @type {(boolean | number)[]} */
	var stack = [];
	var p = stack.pop.bind(stack);
	for (var i = 1; i < range.length; i++) {
		var item = /** @type {SemVerRange | 0 | 1 | 2} */ (range[i]);
		stack.push(
			item == 1
				? p() | p()
				: item == 2
				? p() & p()
				: item
				? satisfy(item, version)
				: !p()
		);
	}
	return !!p();
}
var ensureExistence = function(scopeName, key) {
	var scope = __webpack_require__.S[scopeName];
	if(!scope || !__webpack_require__.o(scope, key)) throw new Error("Shared module " + key + " doesn't exist in shared scope " + scopeName);
	return scope;
};
var findVersion = function(scope, key) {
	var versions = scope[key];
	var key = Object.keys(versions).reduce(function(a, b) {
		return !a || versionLt(a, b) ? b : a;
	}, 0);
	return key && versions[key]
};
var findSingletonVersionKey = function(scope, key) {
	var versions = scope[key];
	return Object.keys(versions).reduce(function(a, b) {
		return !a || (!versions[a].loaded && versionLt(a, b)) ? b : a;
	}, 0);
};
var getInvalidSingletonVersionMessage = function(scope, key, version, requiredVersion) {
	return "Unsatisfied version " + version + " from " + (version && scope[key][version].from) + " of shared singleton module " + key + " (required " + rangeToString(requiredVersion) + ")"
};
var getSingleton = function(scope, scopeName, key, requiredVersion) {
	var version = findSingletonVersionKey(scope, key);
	return get(scope[key][version]);
};
var getSingletonVersion = function(scope, scopeName, key, requiredVersion) {
	var version = findSingletonVersionKey(scope, key);
	if (!satisfy(requiredVersion, version)) warn(getInvalidSingletonVersionMessage(scope, key, version, requiredVersion));
	return get(scope[key][version]);
};
var getStrictSingletonVersion = function(scope, scopeName, key, requiredVersion) {
	var version = findSingletonVersionKey(scope, key);
	if (!satisfy(requiredVersion, version)) throw new Error(getInvalidSingletonVersionMessage(scope, key, version, requiredVersion));
	return get(scope[key][version]);
};
var findValidVersion = function(scope, key, requiredVersion) {
	var versions = scope[key];
	var key = Object.keys(versions).reduce(function(a, b) {
		if (!satisfy(requiredVersion, b)) return a;
		return !a || versionLt(a, b) ? b : a;
	}, 0);
	return key && versions[key]
};
var getInvalidVersionMessage = function(scope, scopeName, key, requiredVersion) {
	var versions = scope[key];
	return "No satisfying version (" + rangeToString(requiredVersion) + ") of shared module " + key + " found in shared scope " + scopeName + ".\n" +
		"Available versions: " + Object.keys(versions).map(function(key) {
		return key + " from " + versions[key].from;
	}).join(", ");
};
var getValidVersion = function(scope, scopeName, key, requiredVersion) {
	var entry = findValidVersion(scope, key, requiredVersion);
	if(entry) return get(entry);
	throw new Error(getInvalidVersionMessage(scope, scopeName, key, requiredVersion));
};
var warn = function(msg) {
	if (typeof console !== "undefined" && console.warn) console.warn(msg);
};
var warnInvalidVersion = function(scope, scopeName, key, requiredVersion) {
	warn(getInvalidVersionMessage(scope, scopeName, key, requiredVersion));
};
var get = function(entry) {
	entry.loaded = 1;
	return entry.get()
};
var init = function(fn) { return function(scopeName, a, b, c) {
	var promise = __webpack_require__.I(scopeName);
	if (promise && promise.then) return promise.then(fn.bind(fn, scopeName, __webpack_require__.S[scopeName], a, b, c));
	return fn(scopeName, __webpack_require__.S[scopeName], a, b, c);
}; };

var load = /*#__PURE__*/ init(function(scopeName, scope, key) {
	ensureExistence(scopeName, key);
	return get(findVersion(scope, key));
});
var loadFallback = /*#__PURE__*/ init(function(scopeName, scope, key, fallback) {
	return scope && __webpack_require__.o(scope, key) ? get(findVersion(scope, key)) : fallback();
});
var loadVersionCheck = /*#__PURE__*/ init(function(scopeName, scope, key, version) {
	ensureExistence(scopeName, key);
	return get(findValidVersion(scope, key, version) || warnInvalidVersion(scope, scopeName, key, version) || findVersion(scope, key));
});
var loadSingleton = /*#__PURE__*/ init(function(scopeName, scope, key) {
	ensureExistence(scopeName, key);
	return getSingleton(scope, scopeName, key);
});
var loadSingletonVersionCheck = /*#__PURE__*/ init(function(scopeName, scope, key, version) {
	ensureExistence(scopeName, key);
	return getSingletonVersion(scope, scopeName, key, version);
});
var loadStrictVersionCheck = /*#__PURE__*/ init(function(scopeName, scope, key, version) {
	ensureExistence(scopeName, key);
	return getValidVersion(scope, scopeName, key, version);
});
var loadStrictSingletonVersionCheck = /*#__PURE__*/ init(function(scopeName, scope, key, version) {
	ensureExistence(scopeName, key);
	return getStrictSingletonVersion(scope, scopeName, key, version);
});
var loadVersionCheckFallback = /*#__PURE__*/ init(function(scopeName, scope, key, version, fallback) {
	if(!scope || !__webpack_require__.o(scope, key)) return fallback();
	return get(findValidVersion(scope, key, version) || warnInvalidVersion(scope, scopeName, key, version) || findVersion(scope, key));
});
var loadSingletonFallback = /*#__PURE__*/ init(function(scopeName, scope, key, fallback) {
	if(!scope || !__webpack_require__.o(scope, key)) return fallback();
	return getSingleton(scope, scopeName, key);
});
var loadSingletonVersionCheckFallback = /*#__PURE__*/ init(function(scopeName, scope, key, version, fallback) {
	if(!scope || !__webpack_require__.o(scope, key)) return fallback();
	return getSingletonVersion(scope, scopeName, key, version);
});
var loadStrictVersionCheckFallback = /*#__PURE__*/ init(function(scopeName, scope, key, version, fallback) {
	var entry = scope && __webpack_require__.o(scope, key) && findValidVersion(scope, key, version);
	return entry ? get(entry) : fallback();
});
var loadStrictSingletonVersionCheckFallback = /*#__PURE__*/ init(function(scopeName, scope, key, version, fallback) {
	if(!scope || !__webpack_require__.o(scope, key)) return fallback();
	return getStrictSingletonVersion(scope, scopeName, key, version);
});
var resolveHandler = function(data) {
	var strict = false
	var singleton = false
	var versionCheck = false
	var fallback = false
	var args = [data.shareScope, data.shareKey];
	if (data.requiredVersion) {
		if (data.strictVersion) strict = true;
		if (data.singleton) singleton = true;
		args.push(parseRange(data.requiredVersion));
		versionCheck = true
	} else if (data.singleton) singleton = true;
	if (data.fallback) {
		fallback = true;
		args.push(data.fallback);
	}
	if (strict && singleton && versionCheck && fallback) return function() { return loadStrictSingletonVersionCheckFallback.apply(null, args); }
	if (strict && versionCheck && fallback) return function() { return loadStrictVersionCheckFallback.apply(null, args); }
	if (singleton && versionCheck && fallback) return function() { return loadSingletonVersionCheckFallback.apply(null, args); }
	if (strict && singleton && versionCheck) return function() { return loadStrictSingletonVersionCheck.apply(null, args); }
	if (singleton && fallback) return function() { return loadSingletonFallback.apply(null, args); }
	if (versionCheck && fallback) return function() { return loadVersionCheckFallback.apply(null, args); }
	if (strict && versionCheck) return function() { return loadStrictVersionCheck.apply(null, args); }
	if (singleton && versionCheck) return function() { return loadSingletonVersionCheck.apply(null, args); }
	if (singleton) return function() { return loadSingleton.apply(null, args); }
	if (versionCheck) return function() { return loadVersionCheck.apply(null, args); }
	if (fallback) return function() { return loadFallback.apply(null, args); }
	return function() { return load.apply(null, args); }
};
var installedModules = {};
__webpack_require__.f.consumes = function(chunkId, promises) {
	var moduleIdToConsumeDataMapping = __webpack_require__.consumesLoadingData.moduleIdToConsumeDataMapping
	var chunkMapping = __webpack_require__.consumesLoadingData.chunkMapping;
	if(__webpack_require__.o(chunkMapping, chunkId)) {
		chunkMapping[chunkId].forEach(function(id) {
			if(__webpack_require__.o(installedModules, id)) return promises.push(installedModules[id]);
			var onFactory = function(factory) {
				installedModules[id] = 0;
				__webpack_require__.m[id] = function(module) {
					delete __webpack_require__.c[id];
					module.exports = factory();
				}
			};
			var onError = function(error) {
				delete installedModules[id];
				__webpack_require__.m[id] = function(module) {
					delete __webpack_require__.c[id];
					throw error;
				}
			};
			try {
				var promise = resolveHandler(moduleIdToConsumeDataMapping[id])();
				if(promise.then) {
					promises.push(installedModules[id] = promise.then(onFactory)['catch'](onError));
				} else onFactory(promise);
			} catch(e) { onError(e); }
		});
	}
}

})();
// webpack/runtime/jsonp_chunk_loading
(() => {

      // object to store loaded and loading chunks
      // undefined = chunk not loaded, null = chunk preloaded/prefetched
      // [resolve, reject, Promise] = chunk loading, 0 = chunk loaded
      var installedChunks = {"444": 0,};
      
        __webpack_require__.f.j = function (chunkId, promises) {
          // JSONP chunk loading for javascript
var installedChunkData = __webpack_require__.o(installedChunks, chunkId)
	? installedChunks[chunkId]
	: undefined;
if (installedChunkData !== 0) {
	// 0 means "already installed".

	// a Promise means "currently loading".
	if (installedChunkData) {
		promises.push(installedChunkData[2]);
	} else {
		if (true) {
			// setup Promise in chunk cache
			var promise = new Promise(function (resolve, reject) {
				installedChunkData = installedChunks[chunkId] = [resolve, reject];
			});
			promises.push((installedChunkData[2] = promise));

			// start chunk loading
			var url = __webpack_require__.p + __webpack_require__.u(chunkId);
			// create error before stack unwound to get useful stacktrace later
			var error = new Error();
			var loadingEnded = function (event) {
				if (__webpack_require__.o(installedChunks, chunkId)) {
					installedChunkData = installedChunks[chunkId];
					if (installedChunkData !== 0) installedChunks[chunkId] = undefined;
					if (installedChunkData) {
						var errorType =
							event && (event.type === 'load' ? 'missing' : event.type);
						var realSrc = event && event.target && event.target.src;
						error.message =
							'Loading chunk ' +
							chunkId +
							' failed.\n(' +
							errorType +
							': ' +
							realSrc +
							')';
						error.name = 'ChunkLoadError';
						error.type = errorType;
						error.request = realSrc;
						installedChunkData[1](error);
					}
				}
			};
			__webpack_require__.l(url, loadingEnded, "chunk-" + chunkId, chunkId);
		} 
	}
}

        }
        // install a JSONP callback for chunk loading
var webpackJsonpCallback = function (parentChunkLoadingFunction, data) {
	var chunkIds = data[0];
	var moreModules = data[1];
	var runtime = data[2];
	// add "moreModules" to the modules object,
	// then flag all "chunkIds" as loaded and fire callback
	var moduleId,
		chunkId,
		i = 0;
	if (chunkIds.some(function (id) { return installedChunks[id] !== 0 })) {
		for (moduleId in moreModules) {
			if (__webpack_require__.o(moreModules, moduleId)) {
				__webpack_require__.m[moduleId] = moreModules[moduleId];
			}
		}
		if (runtime) var result = runtime(__webpack_require__);
	}
	if (parentChunkLoadingFunction) parentChunkLoadingFunction(data);
	for (; i < chunkIds.length; i++) {
		chunkId = chunkIds[i];
		if (
			__webpack_require__.o(installedChunks, chunkId) &&
			installedChunks[chunkId]
		) {
			installedChunks[chunkId][0]();
		}
		installedChunks[chunkId] = 0;
	}
	
};

var chunkLoadingGlobal = self["webpackChunk_anywidget_monorepo"] = self["webpackChunk_anywidget_monorepo"] || [];
chunkLoadingGlobal.forEach(webpackJsonpCallback.bind(null, 0));
chunkLoadingGlobal.push = webpackJsonpCallback.bind(
	null,
	chunkLoadingGlobal.push.bind(chunkLoadingGlobal)
);

})();
/************************************************************************/
// module cache are used so entry inlining is disabled
// startup
// Load entry module and return exports
var __webpack_exports__ = __webpack_require__("95");
(_JUPYTERLAB = typeof _JUPYTERLAB === 'undefined' ? {} : _JUPYTERLAB).anywidget = __webpack_exports__;
})()
;