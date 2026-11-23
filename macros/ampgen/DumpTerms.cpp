// Dump AmpGen amplitudes term by term at phase-space events, as JSON reference values.
#include <complex>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <TRandom3.h>
#include "AmpGen/CoherentSum.h"
#include "AmpGen/EventList.h"
#include "AmpGen/EventType.h"
#include "AmpGen/Generator.h"
#include "AmpGen/MinuitParameterSet.h"
#include "AmpGen/MinuitParameter.h"
#include "AmpGen/OptionsParser.h"
#include "AmpGen/ParticlePropertiesList.h"
#include "AmpGen/PhaseSpace.h"
#include "AmpGen/Property.h"

using namespace AmpGen;
using strings = std::vector<std::string>;

int main(int argc, char **argv) {
  OptionsParser::setArgs(argc, argv);
  EventType eventType(Property<strings>(nullptr, "EventType", {}), Property<bool>(nullptr, "GenerateTimeDependent", false));
  const int nEvents = Property<int>(nullptr, "nEvents", 1000);
  const int seed = Property<int>(nullptr, "Seed", 12345);
  const std::string output = Property<std::string>(nullptr, "Output", "terms.json");
  const strings particles = Property<strings>(nullptr, "Particles", {});

  MinuitParameterSet MPS;
  MPS.loadFromStream();
  CoherentSum sig(eventType, MPS);

  TRandom3 rndm(seed);
  EventList events = Generator<PhaseSpace>(eventType, &rndm).generate(nEvents);
  sig.setEvents(events);
  sig.prepare();

  nlohmann::json out;
  out["event_type"] = eventType.finalStates();
  for (size_t e = 0; e < events.size(); ++e) {
    std::vector<double> p(events[e].address(), events[e].address() + eventType.eventSize());
    out["events"].push_back(p);
    auto total = sig.getValNoCache(events[e]);
    out["total"].push_back({std::real(total), std::imag(total)});
  }
  for (size_t i = 0; i < sig.size(); ++i) {
    nlohmann::json term;
    term["name"] = sig[i].decayDescriptor();
    auto c = sig[i].coupling();
    term["coupling"] = {std::real(c), std::imag(c)};
    for (size_t e = 0; e < events.size(); ++e) {
      auto a = sig[i](events[e])[0];
      term["amp"].push_back({utils::get<0>(a.real()), utils::get<0>(a.imag())});
    }
    out["terms"].push_back(term);
  }
  // AmpGen's default particle properties (options-file overrides are not applied here)
  for (const auto &name : particles) {
    auto props = ParticlePropertiesList::get(name, true);
    if (props == nullptr) { ERROR("Unknown particle " << name); continue; }
    out["particles"][name] = {{"mass", props->mass()}, {"width", props->width()}, {"radius", props->radius()},
                              {"two_spin", props->twoSpin()}, {"parity", props->P()}, {"pdg_id", props->pdgID()},
                              {"charge", props->charge()}};
  }
  std::ofstream os(output);
  os << std::setprecision(17) << out.dump(1) << std::endl;
  INFO("Wrote " << events.size() << " events and " << sig.size() << " terms to " << output);
}
