// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __CPUMineSweeperPlayer__h__
#define __CPUMineSweeperPlayer__h__

#include "MineSweeperPlayerBase.h"
#include <functional>
#include <tuple>
#include <vector>

/**
 */
class CPUMineSweeperPlayer
  : public MineSweeperPlayerBase
{
public:
  CPUMineSweeperPlayer( );
  virtual ~CPUMineSweeperPlayer( ) = default;

  virtual void configure(
    unsigned int w, unsigned int h, unsigned long n
    ) override;
  virtual void play( unsigned int& i, unsigned int& j ) override;
  virtual void report( const unsigned char& c ) override;

private:
  unsigned long long _idx(
    const unsigned int& i, const unsigned int& j
    ) const;

protected:

  using TTuple = std::tuple< double, unsigned int, unsigned int >;

  std::function< bool( const TTuple&, const TTuple& ) > m_Compare;
  std::vector< TTuple > m_Options;
  std::vector< bool > m_Patches;
};

#endif // __CPUMineSweeperPlayer__h__

// eof - $RCSfile$

