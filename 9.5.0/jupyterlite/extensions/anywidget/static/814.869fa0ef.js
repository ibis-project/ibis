"use strict";
(self['webpackChunk_anywidget_monorepo'] = self['webpackChunk_anywidget_monorepo'] || []).push([["814"], {
"148": (function (__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {
var _jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0___namespace_cache;
__webpack_require__.r(__webpack_exports__);
/* harmony import */var _jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(397);
/* harmony import */var _widget_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(30);



/**
 * @typedef JupyterLabRegistry
 * @property {(widget: { name: string, version: string, exports: any }) => void} registerWidget
 */

/* harmony default export */ __webpack_exports__["default"] = ({
	id: "anywidget:plugin",
	requires: [_jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0__.IJupyterWidgetRegistry],
	activate: (
		/** @type {unknown} */ _app,
		/** @type {JupyterLabRegistry} */ registry,
	) => {
		let exports = (0,_widget_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */.Z)(/*#__PURE__*/ (_jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0___namespace_cache || (_jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0___namespace_cache = __webpack_require__.t(_jupyter_widgets_base__WEBPACK_IMPORTED_MODULE_0__, 2))));
		registry.registerWidget({
			name: "anywidget",
			// @ts-expect-error Added by bundler
			version: ("0.9.13"),
			exports,
		});
	},
	autoStart: true,
});


}),
"30": (function (__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {
__webpack_require__.d(__webpack_exports__, {
  Z: function() { return /* export default binding */ __WEBPACK_DEFAULT_EXPORT__; }
});
/* harmony import */var _lukeed_uuid__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(800);
/* harmony import */var solid_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(592);



/**
 * @typedef AnyWidget
 * @prop initialize {import("@anywidget/types").Initialize}
 * @prop render {import("@anywidget/types").Render}
 */

/**
 *  @typedef AnyWidgetModule
 *  @prop render {import("@anywidget/types").Render=}
 *  @prop default {AnyWidget | (() => AnyWidget | Promise<AnyWidget>)=}
 */

/**
 * @param {any} condition
 * @param {string} message
 * @returns {asserts condition}
 */
function assert(condition, message) {
	if (!condition) throw new Error(message);
}

/**
 * @param {string} str
 * @returns {str is "https://${string}" | "http://${string}"}
 */
function is_href(str) {
	return str.startsWith("http://") || str.startsWith("https://");
}

/**
 * @param {string} href
 * @param {string} anywidget_id
 * @returns {Promise<void>}
 */
async function load_css_href(href, anywidget_id) {
	/** @type {HTMLLinkElement | null} */
	let prev = document.querySelector(`link[id='${anywidget_id}']`);

	// Adapted from https://github.com/vitejs/vite/blob/d59e1acc2efc0307488364e9f2fad528ec57f204/packages/vite/src/client/client.ts#L185-L201
	// Swaps out old styles with new, but avoids flash of unstyled content.
	// No need to await the load since we already have styles applied.
	if (prev) {
		let newLink = /** @type {HTMLLinkElement} */ (prev.cloneNode());
		newLink.href = href;
		newLink.addEventListener("load", () => prev?.remove());
		newLink.addEventListener("error", () => prev?.remove());
		prev.after(newLink);
		return;
	}

	return new Promise((resolve) => {
		let link = Object.assign(document.createElement("link"), {
			rel: "stylesheet",
			href,
			onload: resolve,
		});
		document.head.appendChild(link);
	});
}

/**
 * @param {string} css_text
 * @param {string} anywidget_id
 * @returns {void}
 */
function load_css_text(css_text, anywidget_id) {
	/** @type {HTMLStyleElement | null} */
	let prev = document.querySelector(`style[id='${anywidget_id}']`);
	if (prev) {
		// replace instead of creating a new DOM node
		prev.textContent = css_text;
		return;
	}
	let style = Object.assign(document.createElement("style"), {
		id: anywidget_id,
		type: "text/css",
	});
	style.appendChild(document.createTextNode(css_text));
	document.head.appendChild(style);
}

/**
 * @param {string | undefined} css
 * @param {string} anywidget_id
 * @returns {Promise<void>}
 */
async function load_css(css, anywidget_id) {
	if (!css || !anywidget_id) return;
	if (is_href(css)) return load_css_href(css, anywidget_id);
	return load_css_text(css, anywidget_id);
}

/**
 * @param {string} esm
 * @returns {Promise<{ mod: AnyWidgetModule, url: string }>}
 */
async function load_esm(esm) {
	if (is_href(esm)) {
		return {
			mod: await import(/* webpackIgnore: true */ esm),
			url: esm,
		};
	}
	let url = URL.createObjectURL(new Blob([esm], { type: "text/javascript" }));
	let mod = await import(/* webpackIgnore: true */ url);
	URL.revokeObjectURL(url);
	return { mod, url };
}

/** @param {string} anywidget_id */
function warn_render_deprecation(anywidget_id) {
	console.warn(`\
[anywidget] Deprecation Warning for ${anywidget_id}: Direct export of a 'render' will likely be deprecated in the future. To migrate ...

Remove the 'export' keyword from 'render'
-----------------------------------------

export function render({ model, el }) { ... }
^^^^^^

Create a default export that returns an object with 'render'
------------------------------------------------------------

function render({ model, el }) { ... }
         ^^^^^^
export default { render }
                 ^^^^^^

Pin to anywidget>=0.9.0 in your pyproject.toml
----------------------------------------------

dependencies = ["anywidget>=0.9.0"]

To learn more, please see: https://github.com/manzt/anywidget/pull/395.
`);
}

/**
 * @param {string} esm
 * @param {string} anywidget_id
 * @returns {Promise<AnyWidget & { url: string }>}
 */
async function load_widget(esm, anywidget_id) {
	let { mod, url } = await load_esm(esm);
	if (mod.render) {
		warn_render_deprecation(anywidget_id);
		return {
			url,
			async initialize() {},
			render: mod.render,
		};
	}
	assert(
		mod.default,
		`[anywidget] module must export a default function or object.`,
	);
	let widget =
		typeof mod.default === "function" ? await mod.default() : mod.default;
	return { url, ...widget };
}

/**
 * This is a trick so that we can cleanup event listeners added
 * by the user-defined function.
 */
let INITIALIZE_MARKER = Symbol("anywidget.initialize");

/**
 * @param {import("@jupyter-widgets/base").DOMWidgetModel} model
 * @param {unknown} context
 * @return {import("@anywidget/types").AnyModel}
 *
 * Prunes the view down to the minimum context necessary.
 *
 * Calls to `model.get` and `model.set` automatically add the
 * `context`, so we can gracefully unsubscribe from events
 * added by user-defined hooks.
 */
function model_proxy(model, context) {
	return {
		get: model.get.bind(model),
		set: model.set.bind(model),
		save_changes: model.save_changes.bind(model),
		send: model.send.bind(model),
		// @ts-expect-error
		on(name, callback) {
			model.on(name, callback, context);
		},
		off(name, callback) {
			model.off(name, callback, context);
		},
		widget_manager: model.widget_manager,
	};
}

/**
 * @param {void | (() => import('vitest').Awaitable<void>)} fn
 * @param {string} kind
 */
async function safe_cleanup(fn, kind) {
	return Promise.resolve()
		.then(() => fn?.())
		.catch((e) => console.warn(`[anywidget] error cleaning up ${kind}.`, e));
}

/**
 * @template T
 * @typedef {{ data: T, state: "ok" } | { error: any, state: "error" }} Result
 */

/** @type {<T>(data: T) => Result<T>} */
function ok(data) {
	return { data, state: "ok" };
}

/** @type {(e: any) => Result<any>} */
function error(e) {
	return { error: e, state: "error" };
}

/**
 * Cleans up the stack trace at anywidget boundary.
 * You can fully inspect the entire stack trace in the console interactively,
 * but the initial error message is cleaned up to be more user-friendly.
 *
 * @param {unknown} source
 * @returns {never}
 */
function throw_anywidget_error(source) {
	if (!(source instanceof Error)) {
		// Don't know what to do with this.
		throw source;
	}
	let lines = source.stack?.split("\n") ?? [];
	let anywidget_index = lines.findIndex((line) => line.includes("anywidget"));
	let clean_stack =
		anywidget_index === -1 ? lines : lines.slice(0, anywidget_index + 1);
	source.stack = clean_stack.join("\n");
	console.error(source);
	throw source;
}

/**
 * @typedef InvokeOptions
 * @prop {DataView[]} [buffers]
 * @prop {AbortSignal} [signal]
 */

/**
 * @template T
 * @param {import("@anywidget/types").AnyModel} model
 * @param {string} name
 * @param {any} [msg]
 * @param {InvokeOptions} [options]
 * @return {Promise<[T, DataView[]]>}
 */
function invoke(model, name, msg, options = {}) {
	// crypto.randomUUID() is not available in non-secure contexts (i.e., http://)
	// so we use simple (non-secure) polyfill.
	let id = _lukeed_uuid__WEBPACK_IMPORTED_MODULE_0__.v4();
	let signal = options.signal ?? AbortSignal.timeout(3000);

	return new Promise((resolve, reject) => {
		if (signal.aborted) {
			reject(signal.reason);
		}
		signal.addEventListener("abort", () => {
			model.off("msg:custom", handler);
			reject(signal.reason);
		});

		/**
		 * @param {{ id: string, kind: "anywidget-command-response", response: T }} msg
		 * @param {DataView[]} buffers
		 */
		function handler(msg, buffers) {
			if (!(msg.id === id)) return;
			resolve([msg.response, buffers]);
			model.off("msg:custom", handler);
		}
		model.on("msg:custom", handler);
		model.send(
			{ id, kind: "anywidget-command", name, msg },
			undefined,
			options.buffers ?? [],
		);
	});
}

class Runtime {
	/** @type {() => void} */
	#disposer = () => {};
	/** @type {Set<() => void>} */
	#view_disposers = new Set();
	/** @type {import('solid-js').Resource<Result<AnyWidget & { url: string }>>} */
	// @ts-expect-error - Set synchronously in constructor.
	#widget_result;

	/** @param {import("@jupyter-widgets/base").DOMWidgetModel} model */
	constructor(model) {
		this.#disposer = solid_js__WEBPACK_IMPORTED_MODULE_1__/* .createRoot */.so((dispose) => {
			let [css, set_css] = solid_js__WEBPACK_IMPORTED_MODULE_1__/* .createSignal */.gQ(model.get("_css"));
			model.on("change:_css", () => {
				let id = model.get("_anywidget_id");
				console.debug(`[anywidget] css hot updated: ${id}`);
				set_css(model.get("_css"));
			});
			solid_js__WEBPACK_IMPORTED_MODULE_1__/* .createEffect */.GW(() => {
				let id = model.get("_anywidget_id");
				load_css(css(), id);
			});

			/** @type {import("solid-js").Signal<string>} */
			let [esm, setEsm] = solid_js__WEBPACK_IMPORTED_MODULE_1__/* .createSignal */.gQ(model.get("_esm"));
			model.on("change:_esm", async () => {
				let id = model.get("_anywidget_id");
				console.debug(`[anywidget] esm hot updated: ${id}`);
				setEsm(model.get("_esm"));
			});
			/** @type {void | (() => import("vitest").Awaitable<void>)} */
			let cleanup;
			this.#widget_result = solid_js__WEBPACK_IMPORTED_MODULE_1__/* .createResource */.SN(esm, async (update) => {
				await safe_cleanup(cleanup, "initialize");
				try {
					model.off(null, null, INITIALIZE_MARKER);
					let widget = await load_widget(update, model.get("_anywidget_id"));
					cleanup = await widget.initialize?.({
						model: model_proxy(model, INITIALIZE_MARKER),
						experimental: {
							// @ts-expect-error - bind isn't working
							invoke: invoke.bind(null, model),
						},
					});
					return ok(widget);
				} catch (e) {
					return error(e);
				}
			})[0];
			return () => {
				cleanup?.();
				model.off("change:_css");
				model.off("change:_esm");
				dispose();
			};
		});
	}

	/**
	 * @param {import("@jupyter-widgets/base").DOMWidgetView} view
	 * @returns {Promise<() => void>}
	 */
	async create_view(view) {
		let model = view.model;
		let disposer = solid_js__WEBPACK_IMPORTED_MODULE_1__/* .createRoot */.so((dispose) => {
			/** @type {void | (() => import("vitest").Awaitable<void>)} */
			let cleanup;
			let resource = solid_js__WEBPACK_IMPORTED_MODULE_1__/* .createResource */.SN(
				this.#widget_result,
				async (widget_result) => {
					cleanup?.();
					// Clear all previous event listeners from this hook.
					model.off(null, null, view);
					view.$el.empty();
					if (widget_result.state === "error") {
						throw_anywidget_error(widget_result.error);
					}
					let widget = widget_result.data;
					try {
						cleanup = await widget.render?.({
							model: model_proxy(model, view),
							el: view.el,
							experimental: {
								// @ts-expect-error - bind isn't working
								invoke: invoke.bind(null, model),
							},
						});
					} catch (e) {
						throw_anywidget_error(e);
					}
				},
			)[0];
			solid_js__WEBPACK_IMPORTED_MODULE_1__/* .createEffect */.GW(() => {
				if (resource.error) {
					// TODO: Show error in the view?
				}
			});
			return () => {
				dispose();
				cleanup?.();
			};
		});
		// Have the runtime keep track but allow the view to dispose itself.
		this.#view_disposers.add(disposer);
		return () => {
			let deleted = this.#view_disposers.delete(disposer);
			if (deleted) disposer();
		};
	}

	dispose() {
		// biome-ignore lint/complexity/noForEach: Easier to read than a for loop.
		this.#view_disposers.forEach((dispose) => dispose());
		this.#view_disposers.clear();
		this.#disposer();
	}
}

// @ts-expect-error - injected by bundler
let version = ("0.9.13");

/** @param {typeof import("@jupyter-widgets/base")} base */
/* harmony default export */ function __WEBPACK_DEFAULT_EXPORT__({ DOMWidgetModel, DOMWidgetView }) {
	/** @type {WeakMap<AnyModel, Runtime>} */
	let RUNTIMES = new WeakMap();

	class AnyModel extends DOMWidgetModel {
		static model_name = "AnyModel";
		static model_module = "anywidget";
		static model_module_version = version;

		static view_name = "AnyView";
		static view_module = "anywidget";
		static view_module_version = version;

		/** @param {Parameters<InstanceType<DOMWidgetModel>["initialize"]>} args */
		initialize(...args) {
			super.initialize(...args);
			let runtime = new Runtime(this);
			this.once("destroy", () => {
				try {
					runtime.dispose();
				} finally {
					RUNTIMES.delete(this);
				}
			});
			RUNTIMES.set(this, runtime);
		}

		/**
		 * @param {Record<string, any>} state
		 *
		 * We override to support binary trailets because JSON.parse(JSON.stringify())
		 * does not properly clone binary data (it just returns an empty object).
		 *
		 * https://github.com/jupyter-widgets/ipywidgets/blob/47058a373d2c2b3acf101677b2745e14b76dd74b/packages/base/src/widget.ts#L562-L583
		 */
		serialize(state) {
			let serializers =
				/** @type {DOMWidgetModel} */ (this.constructor).serializers || {};
			for (let k of Object.keys(state)) {
				try {
					let serialize = serializers[k]?.serialize;
					if (serialize) {
						state[k] = serialize(state[k], this);
					} else if (k === "layout" || k === "style") {
						// These keys come from ipywidgets, rely on JSON.stringify trick.
						state[k] = JSON.parse(JSON.stringify(state[k]));
					} else {
						state[k] = structuredClone(state[k]);
					}
					if (typeof state[k]?.toJSON === "function") {
						state[k] = state[k].toJSON();
					}
				} catch (e) {
					console.error("Error serializing widget state attribute: ", k);
					throw e;
				}
			}
			return state;
		}
	}

	class AnyView extends DOMWidgetView {
		/** @type {undefined | (() => void)} */
		#dispose = undefined;
		async render() {
			let runtime = RUNTIMES.get(this.model);
			assert(runtime, "[anywidget] runtime not found.");
			assert(!this.#dispose, "[anywidget] dispose already set.");
			this.#dispose = await runtime.create_view(this);
		}
		remove() {
			this.#dispose?.();
			super.remove();
		}
	}

	return { AnyModel, AnyView };
}


}),

}]);
//# sourceMappingURL=814.869fa0ef.js.map