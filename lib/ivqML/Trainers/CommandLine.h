// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Trainers__CommandLine__h__
#define __ivqML__Trainers__CommandLine__h__

namespace ivqML
{
  namespace Trainers
  {
    /**
     */
    template< class _O >
    class CommandLine
      : public _O
    {
    public:
      using Self = CommandLine;
      using Superclass = _O;
      ivqML_Optimizer_Typedefs;

    public:
      CommandLine( );
      virtual ~CommandLine( ) override = default;

      TModel& model( );
      const TModel& model( ) const;

      static bool debug(
        const TScalar& J,
        const TScalar& G,
        const TModel* m,
        const TNatural& i,
        bool d
        );

      virtual void fit( ) override;

    protected:
      virtual void _prepare_training( ) = 0;

    protected:
      TModel m_Model;
      TMatrix m_dX;
      TMatrix m_dY;
      static bool s_Stop;
    };
  } // end namespace
} // end namespace

#include <ivqML/Trainers/CommandLine.hxx>

#endif // __ivqML__Trainers__CommandLine__h__

// eof - $RCSfile$
