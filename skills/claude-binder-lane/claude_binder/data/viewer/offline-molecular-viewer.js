/*
 * Claude Binder offline molecular viewer runtime.
 * Copyright (c) 2026 Claude Binder contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * The page generator supplies the molecular records and scene metadata. This
 * first-party runtime keeps the small method surface used by the generated page
 * in one inline asset, so a downloaded page makes no script or data requests.
 * It uses the $3Dmol.createViewer name and method names inspired by 3Dmol.js;
 * it is not the upstream distribution and does not claim general compatibility.
 */
(function (global) {
  "use strict";

  var BAND_COLORS = {
    vlow: "#fe7d45",
    low: "#ffdb2e",
    high: "#65c9ed",
    vhigh: "#0053b3",
    target: "#707070",
    site: "#ff9c12",
    offsite: "#303030"
  };

  function number(value, fallback) {
    var parsed = Number(value);
    return isFinite(parsed) ? parsed : fallback;
  }

  function atomFromPdb(line) {
    var chain = line.slice(21, 22).trim() || "_";
    var resi = parseInt(line.slice(22, 26).trim(), 10);
    var x = number(line.slice(30, 38), NaN);
    var y = number(line.slice(38, 46), NaN);
    var z = number(line.slice(46, 54), NaN);
    var b = number(line.slice(60, 66), 0);
    var name = line.slice(12, 16).trim();
    var element = line.slice(76, 78).trim() || name.slice(0, 1) || "C";
    if (!isFinite(resi) || !isFinite(x) || !isFinite(y) || !isFinite(z)) return null;
    return { chain: chain, resi: resi, x: x, y: y, z: z, b: b, name: name, element: element };
  }

  function tokenizeCif(line) {
    var tokens = [];
    var match;
    var pattern = /'[^']*'|"[^"]*"|\S+/g;
    while ((match = pattern.exec(line)) !== null) {
      tokens.push(match[0].replace(/^['"]|['"]$/g, ""));
    }
    return tokens;
  }

  function atomFromCif(fields, values) {
    function field(names, fallback) {
      for (var i = 0; i < names.length; i += 1) {
        var index = fields.indexOf(names[i]);
        if (index >= 0) return values[index];
      }
      return fallback;
    }
    var x = number(field(["_atom_site.Cartn_x"], NaN), NaN);
    var y = number(field(["_atom_site.Cartn_y"], NaN), NaN);
    var z = number(field(["_atom_site.Cartn_z"], NaN), NaN);
    var resi = parseInt(field(["_atom_site.auth_seq_id", "_atom_site.label_seq_id"], ""), 10);
    if (!isFinite(x) || !isFinite(y) || !isFinite(z) || !isFinite(resi)) return null;
    return {
      chain: field(["_atom_site.auth_asym_id", "_atom_site.label_asym_id"], "_") || "_",
      resi: resi,
      x: x,
      y: y,
      z: z,
      b: number(field(["_atom_site.B_iso_or_equiv"], 0), 0),
      name: field(["_atom_site.auth_atom_id", "_atom_site.label_atom_id"], "C"),
      element: field(["_atom_site.type_symbol"], "C")
    };
  }

  function parseStructure(text, format) {
    var atoms = [];
    if (String(format).toLowerCase() === "cif") {
      var lines = String(text).split(/\r?\n/);
      var fields = [];
      var inAtomLoop = false;
      for (var i = 0; i < lines.length; i += 1) {
        var line = lines[i].trim();
        if (!line) continue;
        if (line === "loop_") {
          fields = [];
          inAtomLoop = false;
          continue;
        }
        if (line.indexOf("_atom_site.") === 0) {
          fields.push(line.split(/\s+/)[0]);
          inAtomLoop = true;
          continue;
        }
        if (inAtomLoop && fields.length && line.charAt(0) !== "_") {
          var values = tokenizeCif(line);
          if (values.length >= fields.length) {
            var atom = atomFromCif(fields, values);
            if (atom) atoms.push(atom);
          }
        }
      }
      return atoms;
    }
    String(text).split(/\r?\n/).forEach(function (line) {
      if (line.indexOf("ATOM  ") === 0 || line.indexOf("HETATM") === 0) {
        var atom = atomFromPdb(line);
        if (atom) atoms.push(atom);
      }
    });
    return atoms;
  }

  function key(chain, resi) {
    return String(chain) + ":" + String(resi);
  }

  function inList(values, value) {
    return Array.isArray(values) && values.indexOf(Number(value)) !== -1;
  }

  function confidenceColor(value) {
    if (value < 50) return BAND_COLORS.vlow;
    if (value < 70) return BAND_COLORS.low;
    if (value < 90) return BAND_COLORS.high;
    return BAND_COLORS.vhigh;
  }

  function atomColor(atom, scene) {
    if (atom.chain === scene.target_chain) {
      if (inList(scene.epitope, atom.resi)) return BAND_COLORS.site;
      if (inList(scene.offsite, atom.resi)) return BAND_COLORS.offsite;
      return BAND_COLORS.target;
    }
    if (atom.chain === scene.binder_chain) return confidenceColor(atom.b);
    return "#a8a8a8";
  }

  function Viewer(element, options) {
    this.element = element;
    this.options = options || {};
    this.canvas = document.createElement("canvas");
    this.canvas.className = "mol-canvas";
    this.element.appendChild(this.canvas);
    this.models = [];
    this.scene = {};
    this.yaw = 0.55;
    this.pitch = 0.28;
    this.zoom = 1;
    this.dragging = false;
    this.lastX = 0;
    this.lastY = 0;
    this.canvas.addEventListener("pointerdown", this.pointerDown.bind(this));
    this.canvas.addEventListener("pointermove", this.pointerMove.bind(this));
    this.canvas.addEventListener("pointerup", this.pointerUp.bind(this));
    this.canvas.addEventListener("pointerleave", this.pointerUp.bind(this));
    this.canvas.addEventListener("wheel", this.wheel.bind(this), { passive: false });
    this.resizeObserver = typeof ResizeObserver === "function" ? new ResizeObserver(this.resize.bind(this)) : null;
    if (this.resizeObserver) this.resizeObserver.observe(this.element);
    this.resize();
  }

  Viewer.prototype.resize = function () {
    var rect = this.element.getBoundingClientRect();
    var width = Math.max(240, Math.floor(rect.width || 640));
    var height = Math.max(220, Math.floor(rect.height || 520));
    var ratio = global.devicePixelRatio || 1;
    this.canvas.width = Math.floor(width * ratio);
    this.canvas.height = Math.floor(height * ratio);
    this.canvas.style.width = width + "px";
    this.canvas.style.height = height + "px";
    this.render();
  };

  Viewer.prototype.pointerDown = function (event) {
    this.dragging = true;
    this.lastX = event.clientX;
    this.lastY = event.clientY;
    this.canvas.setPointerCapture(event.pointerId);
  };

  Viewer.prototype.pointerMove = function (event) {
    if (!this.dragging) return;
    this.yaw += (event.clientX - this.lastX) * 0.01;
    this.pitch += (event.clientY - this.lastY) * 0.01;
    this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch));
    this.lastX = event.clientX;
    this.lastY = event.clientY;
    this.render();
  };

  Viewer.prototype.pointerUp = function () {
    this.dragging = false;
  };

  Viewer.prototype.wheel = function (event) {
    event.preventDefault();
    this.zoom *= event.deltaY < 0 ? 1.08 : 0.92;
    this.zoom = Math.max(0.25, Math.min(5, this.zoom));
    this.render();
  };

  Viewer.prototype.addModel = function (text, format) {
    var model = { atoms: parseStructure(text, format || "pdb") };
    this.models.push(model);
    return model;
  };

  Viewer.prototype.getModel = function (index) {
    return this.models[index || 0];
  };

  Viewer.prototype.setScene = function (scene) {
    this.scene = scene || {};
    this.render();
  };

  Viewer.prototype.zoomTo = function () {
    this.zoom = 1;
    this.render();
  };

  Viewer.prototype.setBackgroundColor = function (color) {
    this.options.backgroundColor = color;
    this.render();
  };

  Viewer.prototype.render = function () {
    var context = this.canvas.getContext("2d");
    if (!context) return;
    var width = this.canvas.width;
    var height = this.canvas.height;
    var ratio = global.devicePixelRatio || 1;
    context.save();
    context.fillStyle = this.options.backgroundColor || "#ffffff";
    context.fillRect(0, 0, width, height);
    var atoms = [];
    this.models.forEach(function (model) { atoms = atoms.concat(model.atoms); });
    if (!atoms.length) {
      context.fillStyle = "#4b5563";
      context.font = (16 * ratio) + "px sans-serif";
      context.fillText("No atoms were parsed from this structure", 18 * ratio, 30 * ratio);
      context.restore();
      return;
    }
    var center = atoms.reduce(function (sum, atom) {
      return { x: sum.x + atom.x, y: sum.y + atom.y, z: sum.z + atom.z };
    }, { x: 0, y: 0, z: 0 });
    center.x /= atoms.length; center.y /= atoms.length; center.z /= atoms.length;
    var radius = atoms.reduce(function (maximum, atom) {
      return Math.max(maximum, Math.hypot(atom.x - center.x, atom.y - center.y, atom.z - center.z));
    }, 1);
    var sy = Math.sin(this.yaw), cy = Math.cos(this.yaw);
    var sp = Math.sin(this.pitch), cp = Math.cos(this.pitch);
    var projected = atoms.map(function (atom) {
      var x = atom.x - center.x, y = atom.y - center.y, z = atom.z - center.z;
      var x1 = x * cy - z * sy, z1 = x * sy + z * cy;
      var y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      return { atom: atom, x: width / 2 + x1 / radius * width * 0.38 * this.zoom, y: height / 2 - y1 / radius * height * 0.38 * this.zoom, z: z2 };
    }, this);
    projected.sort(function (left, right) { return left.z - right.z; });
    var chains = {};
    projected.forEach(function (point) {
      if (point.atom.name === "CA" || point.atom.name === "C4'") {
        var chain = point.atom.chain;
        if (!chains[chain]) chains[chain] = [];
        chains[chain].push(point);
      }
    });
    Object.keys(chains).forEach(function (chain) {
      var points = chains[chain];
      points.sort(function (left, right) { return left.atom.resi - right.atom.resi; });
      for (var index = 1; index < points.length; index += 1) {
        var previous = points[index - 1];
        var point = points[index];
        context.beginPath();
        context.moveTo(previous.x, previous.y);
        context.lineTo(point.x, point.y);
        var color;
        if (chain === this.scene.binder_chain) color = confidenceColor(point.atom.b);
        else if (inList(this.scene.epitope, point.atom.resi)) color = BAND_COLORS.site;
        else if (inList(this.scene.offsite, point.atom.resi)) color = BAND_COLORS.offsite;
        else color = BAND_COLORS.target;
        context.strokeStyle = color;
        context.globalAlpha = chain === this.scene.binder_chain ? 0.78 : 0.5;
        context.lineWidth = (chain === this.scene.target_chain ? 7 : 5) * ratio;
        context.lineCap = "round";
        context.stroke();
      }
    }, this);
    context.globalAlpha = 1;
    projected.forEach(function (point) {
      var size = (point.atom.name === "CA" || point.atom.name === "C4'") ? 3.4 : 1.6;
      context.beginPath();
      context.fillStyle = atomColor(point.atom, this.scene);
      context.globalAlpha = point.atom.chain === this.scene.target_chain ? 0.8 : 0.72;
      context.arc(point.x, point.y, size * ratio, 0, Math.PI * 2);
      context.fill();
    }, this);
    context.restore();
  };

  global.$3Dmol = {
    createViewer: function (element, options) { return new Viewer(element, options); },
    _parseStructure: parseStructure,
    _colors: BAND_COLORS,
    _residueKey: key
  };
}(window));
